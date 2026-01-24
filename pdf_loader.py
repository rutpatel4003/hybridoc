from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF
import pymupdf.layout
import pymupdf4llm
from streamlit.runtime.uploaded_file_manager import UploadedFile
from config import Config
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import hashlib
from typing import Optional, List, Dict
from table_intelligence import TableExtractor, StructuredTable
import json
import re

TEXT_FILE_EXTENSION = ".txt"
MD_FILE_EXTENSION = '.md'
PDF_FILE_EXTENSION = ".pdf"

_qwen_model = None
_qwen_processor = None

def get_qwen_vl_model():
    """Lazy-load Qwen3-VL-4B-Instruct with 4-bit quantization."""
    global _qwen_model, _qwen_processor

    if _qwen_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        _qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        _qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        _qwen_model.eval()
        if device == "cuda":
            print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return _qwen_model, _qwen_processor

def run_qwen_vl_ocr(image_bytes: bytes) -> str:
    """Extract text, tables, and formulas using Qwen3-VL-4B-Instruct."""
    try:
        model, processor = get_qwen_vl_model()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if image.size[0] < 10 or image.size[1] < 10:
            return ""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text, tables, and mathematical formulas from this image. For tables, output clean markdown format with | separators. For math, use LaTeX notation. Preserve all numerical values exactly as shown."}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text.strip()

    except Exception as e:
        import traceback
        print(f"Qwen-VL OCR failed: {e}")
        print(traceback.format_exc())
        return ""

@dataclass
class ContentBlock:
    """Represents a block of content from PDF"""
    content: str
    content_type: str  # 'text', 'table', 'figure'
    page_num: int
    bbox: Optional[tuple] = None
    table_data: Optional[StructuredTable] = None

def extract_pdf_content_with_structure(data: bytes, use_got_ocr: bool = True) -> List[ContentBlock]:
    """
    Extract content from PDF with structure detection.
    """
    with fitz.open(stream=data, filetype="pdf") as doc:
        all_blocks: List[ContentBlock] = []
        table_extractor = TableExtractor()

        # unified caption regex for BOTH tables and figures
        caption_re = re.compile(
            r"(table|tab\.|figure|fig\.)\s+\d+(?:\.\d+)?(?:\s*[:.\-–—]\s*.+)?",
            re.IGNORECASE
        )

        def _extract_caption_from_image(page, bbox):
            """
            OCR a small strip above the table/figure bbox to catch captions
            like 'Table 2: ...' even if caption is embedded in the image.
            """
            if not bbox:
                return ""

            x0, y0, x1, y1 = bbox
            # search a small band above the table
            cap_height = getattr(Config.Preprocessing, "IMAGE_TABLE_CAPTION_HEIGHT", 140)
            cap_y0 = max(0, y0 - cap_height)
            cap_bbox = (x0, cap_y0, x1, y0)
            try:
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=cap_bbox)
                text = run_qwen_vl_ocr(pix.tobytes("png"))

                for line in text.splitlines():
                    candidate = " ".join(line.strip().split())
                    if caption_re.search(candidate):
                        return candidate
            except Exception as e:
                print(f"Caption OCR failed: {e}")
            return ""

        for page_num, page in enumerate(doc):
            print(f"Processing page {page_num + 1}/{len(doc)}...")

            # one pass: get layout blocks in reading order
            layout_blocks = page.get_text("dict", sort=True).get("blocks", [])

            page_blocks: List[ContentBlock] = []

            # native tables 
            found_tables = page.find_tables()
            table_bboxes: List[tuple] = []

            for table in found_tables:
                header = table.header.names
                if not header:
                    header = [f"Col {i+1}" for i in range(table.col_count)]
                else:
                    # convert None values to strings
                    header = [str(h) if h is not None else f"Col {i+1}"
                              for i, h in enumerate(header)]

                rows = []
                for row_data in table.extract():
                    if row_data == table.header.names:
                        continue
                    cleaned_row = [str(cell) if cell is not None else "" for cell in row_data]
                    if len(cleaned_row) == len(header):
                        rows.append(dict(zip(header, cleaned_row)))

                # markdown representation
                markdown_lines = [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(["---"] * len(header)) + " |",
                ]
                for row in rows:
                    row_values = [str(row.get(h, "")) for h in header]
                    markdown_lines.append("| " + " | ".join(row_values) + " |")
                raw_markdown = "\n".join(markdown_lines)

                structured_table = StructuredTable(
                    headers=header,
                    rows=rows,
                    raw_markdown=raw_markdown,
                    num_rows=len(rows),
                    num_cols=len(header),
                )

                tbbox = tuple(table.bbox)
                table_bboxes.append(tbbox)

                page_blocks.append(
                    ContentBlock(
                        content=raw_markdown,
                        content_type="table",
                        page_num=page_num + 1,
                        bbox=tbbox,
                        table_data=structured_table,
                    )
                )

            def _overlaps_table(bbox: tuple) -> bool:
                if not bbox or bbox == (0, 0, 0, 0):
                    return False
                b = fitz.Rect(bbox)
                for t_bbox in table_bboxes:
                    if b.intersects(fitz.Rect(t_bbox)):
                        return True
                return False

            # Text + image blocks (OCR for image blocks) 
            for block in layout_blocks:
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))

                if _overlaps_table(bbox):
                    continue

                if block.get("type") == 0:
                    text_content = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_content.append(span.get("text", ""))
                    content = "\n".join([t for t in text_content if t is not None])

                    if content.strip():
                        page_blocks.append(
                            ContentBlock(
                                content=content,
                                content_type="text",
                                page_num=page_num + 1,
                                bbox=bbox,
                            )
                        )

                elif block.get("type") == 1 and use_got_ocr:
                    try:
                        if bbox:
                            mat = fitz.Matrix(2, 2)
                            pix = page.get_pixmap(matrix=mat, clip=bbox)
                            image_bytes = pix.tobytes("png")

                            extracted_text = run_qwen_vl_ocr(image_bytes)

                            if extracted_text:
                                table_result = table_extractor.extract_table_from_ocr(extracted_text)
                                if table_result:
                                    structured_table, original_text = table_result

                                    # OCR caption from image if enabled
                                    if getattr(Config.Preprocessing, "ENABLE_IMAGE_TABLE_CAPTION_OCR", True):
                                        caption_text = _extract_caption_from_image(page, bbox)
                                        if caption_text and caption_text not in original_text:
                                            original_text = f"{caption_text}\n{original_text}"
                                    page_blocks.append(
                                        ContentBlock(
                                            content=original_text,
                                            content_type="table",
                                            page_num=page_num + 1,
                                            bbox=bbox,
                                            table_data=structured_table,
                                        )
                                    )
                                else:
                                    page_blocks.append(
                                        ContentBlock(
                                            content=extracted_text,
                                            content_type="figure",
                                            page_num=page_num + 1,
                                            bbox=bbox,
                                        )
                                    )
                    except Exception as e:
                        print(f"Failed to process image on page {page_num + 1}: {e}")

            # attach captions to nearest table 
            caption_block_ids_to_drop = set()
            table_blocks = [b for b in page_blocks if b.content_type == "table" and b.bbox]
            text_blocks = [b for b in page_blocks if b.content_type == "text" and b.bbox]
            for tb in table_blocks:
                trect = fitz.Rect(tb.bbox)
                best = None  # (distance, text_block)

                for tx in text_blocks:
                    if id(tx) in caption_block_ids_to_drop:
                        continue
                    candidate = " ".join(tx.content.strip().split())
                    match = caption_re.search(candidate)
                    if not match:
                        continue

                    caption_text = candidate[match.start():].strip()
                    txrect = fitz.Rect(tx.bbox)
                    # must be reasonably near vertically
                    if txrect.y0 >= trect.y1:
                        dist = txrect.y0 - trect.y1  # below
                    else:
                        dist = trect.y0 - txrect.y1  # above
                    if dist < 0 or dist > 200:
                        continue
                    x_overlap = max(0.0, min(trect.x1, txrect.x1) - max(trect.x0, txrect.x0))
                    if x_overlap < (min(trect.width, txrect.width) * 0.1):
                        continue

                    if best is None or dist < best[0]:
                        best = (dist, tx)

                if best:
                    _, cap_block = best
                    caption_text = cap_block.content.strip()
                    if caption_text and caption_text not in tb.content:
                        tb.content = f"{caption_text}\n{tb.content}"
                    caption_block_ids_to_drop.add(id(cap_block))

            page_blocks = [b for b in page_blocks if id(b) not in caption_block_ids_to_drop]
            # keep reading order 
            def _sort_key(b: ContentBlock):
                if b.bbox:
                    x0, y0, x1, y1 = b.bbox
                    return (y0, x0)
                return (float("inf"), float("inf"))

            page_blocks.sort(key=_sort_key)
            all_blocks.extend(page_blocks)

        return all_blocks


def format_content_blocks_as_text(content_blocks: List[ContentBlock]) -> str:
    """
    Convert structured content blocks back to text format for storage.

    Uses markers compatible with data_ingestor.py:
      [TABLE: ...] ... [/TABLE]
      [FIGURE] ... [/FIGURE]
    """
    output: List[str] = []

    for block in content_blocks:
        output.append(f"\n--- PAGE {block.page_num} ---\n")

        if block.content_type == "table":
            output.append("[TABLE:]")

            # caption + markdown table (if caption was attached)
            if block.content and block.content.strip():
                output.append(block.content.strip())

            # extra BM25-friendly searchable table text
            if block.table_data:
                output.append("")
                output.append(block.table_data.to_searchable_text())

            output.append("[/TABLE]")
            output.append("")
            continue

        if block.content_type == "figure":
            output.append("[FIGURE]")
            output.append((block.content or "").strip())
            output.append("[/FIGURE]")
            output.append("")
            continue

        output.append((block.content or "").strip())
        output.append("")

    return "\n".join(output)


def extract_pdf_content_with_pymupdf4llm(data: bytes) -> tuple[str, List[ContentBlock]]:
    """
    Hybrid extraction:
    - pymupdf4llm for text (better context preservation and reading order)
    - Qwen3-VL-4B for tables (accurate handling of complex formatting, superscripts, etc.)
    """
    content_blocks: List[ContentBlock] = []
    table_extractor = TableExtractor()

    with fitz.open(stream=data, filetype="pdf") as doc:
        total_pages = len(doc)
        print(f"📄 Extracting text from {total_pages} pages using pymupdf4llm...")

        # Step 1: Extract text with pymupdf4llm (skip tables for now)
        md_text = pymupdf4llm.to_markdown(
            doc,
            page_chunks=False,
            write_images=False,
            show_progress=False,
            margins=(0, 50, 0, 50),
        )
        print(f"✓ Text extraction complete")

        # Step 2: Parse text blocks only (mark table locations)
        print(f"📝 Parsing text blocks and marking table locations...")
        page_sections = re.split(r'\n-----\n\n## Page \d+\n', md_text)
        table_placeholder_count = 0

        for page_num, section in enumerate(page_sections, start=1):
            if not section.strip():
                continue

            lines = section.split('\n')
            current_block = []

            for line in lines:
                if line.strip().startswith('|'):
                    # flush text before table
                    if current_block:
                        text_content = '\n'.join(current_block)
                        if text_content.strip():
                            content_blocks.append(ContentBlock(
                                content=text_content,
                                content_type="text",
                                page_num=page_num,
                                bbox=None
                            ))
                        current_block = []

                    # mark table location with placeholder
                    content_blocks.append(ContentBlock(
                        content=f"__TABLE_PLACEHOLDER_{table_placeholder_count}__",
                        content_type="table_placeholder",
                        page_num=page_num,
                        bbox=None
                    ))
                    table_placeholder_count += 1
                elif line.strip():
                    current_block.append(line)
                elif current_block:
                    text_content = '\n'.join(current_block)
                    if text_content.strip():
                        content_blocks.append(ContentBlock(
                            content=text_content,
                            content_type="text",
                            page_num=page_num,
                            bbox=None
                        ))
                    current_block = []

            if current_block:
                text_content = '\n'.join(current_block)
                if text_content.strip():
                    content_blocks.append(ContentBlock(
                        content=text_content,
                        content_type="text",
                        page_num=page_num,
                        bbox=None
                    ))

        # Step 3: Use Qwen3-VL to extract actual tables
        print(f"🔍 Scanning for tables/figures with Qwen3-VL...")
        extracted_tables = []
        tables_found = 0
        figures_found = 0

        for page_num, page in enumerate(doc, start=1):
            print(f"  Page {page_num}/{total_pages}: Processing tables/figures...", end='\r')
            layout_blocks = page.get_text("dict", sort=True).get("blocks", [])

            for block in layout_blocks:
                if block.get("type") == 0:  # text block
                    bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                    text_lines = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        text_lines.append(line_text)

                    block_text = "\n".join(text_lines)

                    # detect table-like structure
                    if _is_table_heuristic(block_text):
                        try:
                            mat = fitz.Matrix(2, 2)
                            pix = page.get_pixmap(matrix=mat, clip=bbox)
                            image_bytes = pix.tobytes("png")

                            extracted_text = run_qwen_vl_ocr(image_bytes)

                            if extracted_text and len(extracted_text) > 30:
                                table_result = table_extractor.extract_table_from_ocr(extracted_text)
                                if table_result:
                                    structured_table, original_text = table_result
                                    extracted_tables.append(ContentBlock(
                                        content=original_text,
                                        content_type="table",
                                        page_num=page_num,
                                        bbox=bbox,
                                        table_data=structured_table
                                    ))
                                    tables_found += 1
                        except Exception as e:
                            print(f"\n⚠ Table OCR failed on page {page_num}: {e}")

                elif block.get("type") == 1:  # image block
                    try:
                        bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                        if bbox and bbox != (0, 0, 0, 0):
                            mat = fitz.Matrix(2, 2)
                            pix = page.get_pixmap(matrix=mat, clip=bbox)
                            image_bytes = pix.tobytes("png")

                            extracted_text = run_qwen_vl_ocr(image_bytes)

                            if extracted_text and len(extracted_text) > 50:
                                table_result = table_extractor.extract_table_from_ocr(extracted_text)
                                if table_result:
                                    structured_table, original_text = table_result
                                    extracted_tables.append(ContentBlock(
                                        content=original_text,
                                        content_type="table",
                                        page_num=page_num,
                                        bbox=bbox,
                                        table_data=structured_table
                                    ))
                                    tables_found += 1
                                else:
                                    content_blocks.append(ContentBlock(
                                        content=extracted_text,
                                        content_type="figure",
                                        page_num=page_num,
                                        bbox=bbox
                                    ))
                                    figures_found += 1
                    except Exception as e:
                        print(f"\n⚠ Image OCR failed on page {page_num}: {e}")

        print(f"\n✓ OCR complete: Found {tables_found} tables, {figures_found} figures")

        # Step 4: Replace placeholders with actual tables
        table_idx = 0
        final_blocks = []
        for block in content_blocks:
            if block.content_type == "table_placeholder" and table_idx < len(extracted_tables):
                final_blocks.append(extracted_tables[table_idx])
                table_idx += 1
            elif block.content_type != "table_placeholder":
                final_blocks.append(block)

        # add any remaining tables not matched to placeholders
        final_blocks.extend(extracted_tables[table_idx:])

        print(f"✓ Hybrid extraction complete: {len(final_blocks)} content blocks created")

        return md_text, final_blocks


def _is_table_heuristic(text: str) -> bool:
    """Detect if text block contains table-like structure."""
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return False

    has_multiple_numbers = sum(1 for line in lines if sum(c.isdigit() for c in line) > 2) > len(lines) * 0.4
    has_tabs_or_spacing = sum(1 for line in lines if '\t' in line or '  ' * 3 in line) > len(lines) * 0.3
    short_dense_lines = sum(1 for line in lines if 10 < len(line.strip()) < 100) > len(lines) * 0.5

    return (has_tabs_or_spacing or has_multiple_numbers) and short_dense_lines


def extract_pdf_content(data: bytes, use_got_ocr: bool = True) -> str:
    """
    Extract content from PDF with enhanced table detection.
    Uses pymupdf4llm for text extraction and Qwen3-VL-4B for table/figure OCR.
    """
    if getattr(Config.Preprocessing, 'USE_PYMUPDF4LLM', True):
        # use pymupdf4llm for better table and context handling
        md_text, content_blocks = extract_pdf_content_with_pymupdf4llm(data)

        # if GOT-OCR is enabled, process images/figures separately
        if use_got_ocr:
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page_num, page in enumerate(doc, start=1):
                    # find image blocks
                    layout_blocks = page.get_text("dict", sort=True).get("blocks", [])
                    for block in layout_blocks:
                        if block.get("type") == 1:  # image block
                            try:
                                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                                if bbox and bbox != (0, 0, 0, 0):
                                    mat = fitz.Matrix(2, 2)
                                    pix = page.get_pixmap(matrix=mat, clip=bbox)
                                    image_bytes = pix.tobytes("png")

                                    extracted_text = run_qwen_vl_ocr(image_bytes)

                                    if extracted_text and len(extracted_text) > 50:
                                        # check if it's a table or figure
                                        table_result = TableExtractor().extract_table_from_ocr(extracted_text)
                                        if table_result:
                                            structured_table, original_text = table_result
                                            content_blocks.append(ContentBlock(
                                                content=original_text,
                                                content_type="table",
                                                page_num=page_num,
                                                bbox=bbox,
                                                table_data=structured_table
                                            ))
                                        else:
                                            content_blocks.append(ContentBlock(
                                                content=extracted_text,
                                                content_type="figure",
                                                page_num=page_num,
                                                bbox=bbox
                                            ))
                            except Exception as e:
                                print(f"OCR failed for image on page {page_num}: {e}")

        return format_content_blocks_as_text(content_blocks)
    else:
        # fallback to original method
        content_blocks = extract_pdf_content_with_structure(data, use_got_ocr)
        return format_content_blocks_as_text(content_blocks)

@dataclass 
class File:
    name: str
    content: str
    content_blocks: Optional[List[ContentBlock]] = None  # store structured version


def load_uploaded_file(uploaded_file: UploadedFile) -> File:
    """
    Load and process uploaded file with persistent disk caching.
    Enhanced with table intelligence.
    """
    file_extension = Path(uploaded_file.name).suffix
    
    if file_extension not in Config.ALLOWED_FILE_EXTENSIONS:
        raise ValueError(
            f"Invalid file extension: {file_extension} for file {uploaded_file.name}"
        )
    
    # get raw bytes to generate a unique hash
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    # define cache directory and file path
    cache_dir = Config.Path.DATA_DIR / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{file_hash}.txt"
    cache_blocks_path = cache_dir / f"{file_hash}_blocks.json"

    # load from disk if file is processed already
    if cache_path.exists():
        print(f"Cache hit: Loading processed text for {uploaded_file.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # try to load structured blocks if available
        content_blocks = None
        if cache_blocks_path.exists():
            try:
                with open(cache_blocks_path, "r", encoding="utf-8") as f:
                    blocks_data = json.load(f)
                    # reconstruct ContentBlock objects
                    content_blocks = []
                    for block_dict in blocks_data:
                        table_data = None
                        if block_dict.get('table_data'):
                            td = block_dict['table_data']
                            table_data = StructuredTable(
                                headers=td['headers'],
                                rows=td['rows'],
                                raw_markdown=td['raw_markdown'],
                                num_rows=td['num_rows'],
                                num_cols=td['num_cols']
                            )
                        
                        content_blocks.append(ContentBlock(
                            content=block_dict['content'],
                            content_type=block_dict['content_type'],
                            page_num=block_dict['page_num'],
                            bbox=tuple(block_dict['bbox']) if block_dict.get('bbox') else None,
                            table_data=table_data
                        ))
            except Exception as e:
                print(f"Warning: Could not load structured blocks: {e}")
        
        return File(name=uploaded_file.name, content=content, content_blocks=content_blocks)
    
    # if not in cache, run the extraction
    print(f"\n{'='*60}")
    print(f"📥 Processing: {uploaded_file.name}")
    print(f"{'='*60}")

    if file_extension == PDF_FILE_EXTENSION:
        print(f"🔧 Mode: Native PyMuPDF + Qwen3-VL-4B OCR")
        content_blocks = extract_pdf_content_with_structure(file_bytes, use_got_ocr=True)
        content = format_content_blocks_as_text(content_blocks)

        print(f"\n💾 Caching results for future use...")
        # Save structured blocks to cache
        try:
            blocks_data = []
            for block in content_blocks:
                block_dict = {
                    'content': block.content,
                    'content_type': block.content_type,
                    'page_num': block.page_num,
                    'bbox': list(block.bbox) if block.bbox else None,
                    'table_data': block.table_data.to_dict() if block.table_data else None
                }
                blocks_data.append(block_dict)

            with open(cache_blocks_path, "w", encoding="utf-8") as f:
                json.dump(blocks_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not cache structured blocks: {e}")
        
    else:
        content = file_bytes.decode("utf-8")
        content_blocks = None
    
    # save the result to avoid running OCR again
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Processing complete!")
    print(f"{'='*60}\n")

    return File(name=uploaded_file.name, content=content, content_blocks=content_blocks)


# cleanup function to free GPU memory when done
def cleanup_ocr_model():
    """Free GPU memory after processing."""
    global _qwen_model, _qwen_processor

    if _qwen_model is not None:
        del _qwen_model
        del _qwen_processor
        _qwen_model = None
        _qwen_processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Qwen-VL model unloaded, GPU memory freed")