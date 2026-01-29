# pdf_loader.py - PART 1/2
# Changes: Enhanced caption detection, table validation, deduplication

from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF
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

# model loading functions (unchanged)
def get_qwen_vl_model():
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        quantization_config = None
        use_quantization = False
        
        try:
            import importlib.metadata
            importlib.metadata.version("bitsandbytes")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            use_quantization = True
            print("Using 4-bit quantization with bitsandbytes")
        except Exception as e:
            print(f"bitsandbytes not available ({e}), loading model without quantization")

        if use_quantization:
            _qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-4B-Instruct",
                quantization_config=quantization_config,
                device_map="auto",
                dtype=torch.float16 if device == "cuda" else torch.float32,
                attn_implementation="flash_attention_2",
            )
        else:
            _qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-4B-Instruct",
                device_map="auto",
                dtype=torch.float16 if device == "cuda" else torch.float32,
                attn_implementation="flash_attention_2",
            )

        _qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        _qwen_model.eval()
        if device == "cuda":
            print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return _qwen_model, _qwen_processor

def unload_qwen_vl_model():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        print("Unloading Qwen VL model...")
        del _qwen_model
        _qwen_model = None
    if _qwen_processor is not None:
        del _qwen_processor
        _qwen_processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"VRAM after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def run_qwen_vl_ocr(image_bytes: bytes) -> str:
    try:
        model, processor = get_qwen_vl_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.size[0] < 10 or image.size[1] < 10:
            return ""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": """Extract all content from this image precisely.

FOR TABLES:
- Output as markdown table with | separators
- CRITICAL: Preserve EXACT column structure - count columns from the header row
- If a cell is empty/missing, use empty cell (||) - do NOT skip or merge columns
- Keep ALL columns even if some rows have missing values
- Each row must have the SAME number of columns as the header
- Include the separator row (|---|---|...) after headers

FOR MATH/FORMULAS:
- Use LaTeX notation (e.g., $10^{18}$)

FOR TEXT:
- Preserve exact numbers and values as shown
- Keep formatting and structure"""}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

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
    content: str
    content_type: str
    page_num: int
    bbox: Optional[tuple] = None
    table_data: Optional[StructuredTable] = None
    caption_label: Optional[str] = None
    section_header: Optional[str] = None

    def __post_init__(self):
        """Normalize OCR artifacts in content."""
        from table_intelligence import normalize_text
        if self.content:
            self.content = normalize_text(self.content)

@dataclass 
class File:
    name: str
    content: str
    content_blocks: Optional[List[ContentBlock]] = None  # store structured version
    
CAPTION_PATTERN = re.compile(
    r'^\s*(Table|Figure|Fig\.?)\s+(\d+(?:\.\d+)?)(?:[:.]\s*)?([^\n]{0,150})?',
    re.IGNORECASE | re.MULTILINE
)

SECTION_HEADER_PATTERN = re.compile(
    r'^(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?[A-Z][A-Za-z\s]+$',
    re.MULTILINE
)

def find_caption_regions(page, page_text: str) -> List[Dict]:
    """Caption detection with fallback for unlocalizable captions"""
    regions = []
    seen_labels = set()
    
    for match in CAPTION_PATTERN.finditer(page_text):
        label_type = match.group(1).strip()
        label_num = match.group(2).strip()
        caption_desc = match.group(3) or ""
        
        caption_label = f"{label_type} {label_num}"
        normalized_label = caption_label.lower().replace(' ', '').replace('.', '')
        
        if normalized_label in seen_labels:
            continue
        seen_labels.add(normalized_label)
        
        content_type = 'table' if 'table' in label_type.lower() else 'figure'
        text_instances = page.search_for(caption_label)
        
        if text_instances:
            bbox = text_instances[0]
            full_caption = f"{caption_label}{': ' + caption_desc if caption_desc.strip() else ''}"
            regions.append({
                'type': content_type,
                'label': caption_label,
                'caption': full_caption.strip() if full_caption.strip() else caption_label,
                'caption_bbox': tuple(bbox),
                'page_num': page.number + 1
            })
        else:
            # fallback: use heuristic bbox
            print(f"Caption '{caption_label}' found in text but not on page - using heuristic bbox")
            page_rect = page.rect
            heuristic_bbox = (50, page_rect.height * 0.3, page_rect.width - 50, page_rect.height * 0.7)
            full_caption = f"{caption_label}{': ' + caption_desc if caption_desc.strip() else ''}"
            regions.append({
                'type': content_type,
                'label': caption_label,
                'caption': full_caption.strip() if full_caption.strip() else caption_label,
                'caption_bbox': heuristic_bbox,
                'page_num': page.number + 1
            })
    
    return regions

def expand_region_to_content(page, caption_bbox: tuple, content_type: str) -> tuple:
    x0, y0, x1, y1 = caption_bbox
    page_height = page.rect.height
    page_width = page.rect.width
    expand_x = 50
    new_x0 = max(0, x0 - expand_x)
    new_x1 = min(page_width, x1 + expand_x)
    expand_height = getattr(Config.Preprocessing, 'TABLE_FIGURE_REGION_HEIGHT', 500)
    
    if content_type == 'table':
        return (new_x0, max(0, y0 - 30), new_x1, min(page_height, y1 + expand_height))
    else:
        return (new_x0, max(0, y0 - expand_height), new_x1, min(page_height, y1 + 50))

def extract_section_headers(page_text: str) -> List[str]:
    headers = []
    lines = page_text.split('\n')
    for line in lines:
        line = line.strip()
        if re.match(r'^(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)[A-Z]', line):
            if len(line) < 80:
                headers.append(line)
        elif line.isupper() and 3 < len(line) < 50:
            headers.append(line)
    return headers

# comprehensive table validation to prevent figure grids
def _is_valid_table(table, page, page_text: str) -> tuple[bool, str]:
    """Validate if PyMuPDF table is real (not figure grid)"""
    try:
        rows_data = table.extract()
    except:
        return False, "extraction_failed"
    
    if table.col_count < 2 or len(rows_data) < 2:
        return False, "insufficient_dimensions"
    
    # check text content
    text_cells = 0
    total_cells = 0
    numeric_cells = 0
    
    for row in rows_data:
        for cell in row:
            total_cells += 1
            if cell and str(cell).strip():
                text_cells += 1
                if re.search(r'\d', str(cell)):
                    numeric_cells += 1
    
    if total_cells == 0:
        return False, "no_cells"
    
    text_ratio = text_cells / total_cells
    if text_ratio < 0.3:
        return False, f"low_text_ratio_{text_ratio:.2f}"
    
    # detect figure layout grids
    if table.col_count <= 3 and table.row_count <= 3:
        bbox = tuple(table.bbox)
        x0, y0, x1, y1 = bbox
        expanded_bbox = (max(0, x0-50), max(0, y0-100), min(page.rect.width, x1+50), min(page.rect.height, y1+100))
        nearby_text = page.get_text(clip=expanded_bbox).lower()
        
        figure_indicators = [
            r'figure\s+\d+', r'fig\.\s*\d+', r'\(a\)\s*\(b\)', r'\(a\).*\(b\).*\(c\)',
            r'subfigure', r'diagram', r'architecture'
        ]
        
        for pattern in figure_indicators:
            if re.search(pattern, nearby_text):
                return False, f"figure_indicator_{pattern}"
        
        if text_ratio < 0.5 and numeric_cells < 2:
            return False, "likely_figure_layout"
    
    # check meaningful headers
    if rows_data:
        first_row = rows_data[0]
        header_text_count = sum(1 for cell in first_row if cell and str(cell).strip() and len(str(cell).strip()) > 1)
        if header_text_count < 2:
            return False, "weak_headers"
    
    # check for actual text in bbox
    table_region_text = page.get_text(clip=tuple(table.bbox)).strip()
    if len(table_region_text) < 20:
        return False, "insufficient_text_in_region"
    
    # larger tables should have numeric content or variety
    if table.col_count > 3 or table.row_count > 3:
        if numeric_cells == 0 and text_cells > 10:
            unique_values = set()
            for row in rows_data:
                for cell in row:
                    if cell and str(cell).strip():
                        unique_values.add(str(cell).strip())
            if len(unique_values) < 3:
                return False, "low_information_content"
    
    return True, "valid"

def detect_table_in_text_block(block_text: str) -> bool:
    """Strict heuristic - same as original"""
    lines = block_text.strip().split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    if len(non_empty_lines) < 4:
        return False
    
    code_patterns = ['def ', 'class ', 'import ', 'return ', 'for ', 'if ', '= ', '==', '!=', '+=', '()', '{}', '[]']
    code_indicators = sum(1 for p in code_patterns if p in block_text)
    if code_indicators >= 2:
        return False
    
    math_symbols = ['∑', '∏', '∫', '√', '≤', '≥', '→', '←', 'softmax', 'argmax', 'log ', 'exp ']
    if any(sym in block_text for sym in math_symbols):
        return False
    
    pipe_lines = sum(1 for line in lines if '|' in line)
    has_separator = any(re.search(r'\|[\s]*[-:]+[\s]*\|', line) or re.match(r'^\s*\|(\s*-+\s*\|)+\s*$', line) for line in lines)
    if pipe_lines >= 4 and has_separator:
        return True
    
    col_counts = []
    for line in non_empty_lines:
        parts = re.split(r'[\t]|\s{3,}', line.strip())
        parts = [p for p in parts if p.strip()]
        if len(parts) > 1:
            col_counts.append(len(parts))
    
    if len(col_counts) >= 4:
        most_common = max(set(col_counts), key=col_counts.count)
        consistency = col_counts.count(most_common) / len(col_counts)
        if most_common >= 3 and consistency > 0.7:
            return True
    
    return False


# deduplication function
def _merge_duplicate_tables(all_blocks: List[ContentBlock]) -> List[ContentBlock]:
    """Merge duplicate table extractions by content similarity"""
    from difflib import SequenceMatcher
    tables = [b for b in all_blocks if b.content_type == 'table']
    non_tables = [b for b in all_blocks if b.content_type != 'table']
    
    if len(tables) <= 1:
        return all_blocks
    
    tables_by_page = {}
    for t in tables:
        tables_by_page.setdefault(t.page_num, []).append(t)
    
    deduplicated_tables = []
    
    for page_num, page_tables in tables_by_page.items():
        if len(page_tables) == 1:
            deduplicated_tables.append(page_tables[0])
            continue
        kept = []
        for t1 in page_tables:
            is_duplicate = False
            for t2 in kept:
                similarity = SequenceMatcher(None, t1.content[:500], t2.content[:500]).ratio()
                if similarity > 0.7:
                    is_duplicate = True
                    if t1.table_data and not t2.table_data:
                        kept.remove(t2)
                        kept.append(t1)
                    elif t1.table_data and t2.table_data:
                        if t1.table_data.num_rows > t2.table_data.num_rows:
                            kept.remove(t2)
                            kept.append(t1)
                    break
            
            if not is_duplicate:
                kept.append(t1)
        
        deduplicated_tables.extend(kept)
        if len(kept) < len(page_tables):
            print(f"    Page {page_num}: Merged {len(page_tables)} → {len(kept)} tables (removed duplicates)")
    
    return non_tables + deduplicated_tables


def _region_overlaps_any(page_num: int, bbox: tuple, detected_regions: List[tuple], threshold: float = 0.5) -> bool:
    """Same as original"""
    if not bbox or bbox == (0, 0, 0, 0):
        return False
    
    b = fitz.Rect(bbox)
    for (pn, region_bbox) in detected_regions:
        if pn != page_num:
            continue
        r = fitz.Rect(region_bbox)
        intersection = b & r
        if intersection.is_empty:
            continue
        overlap_area = intersection.width * intersection.height
        b_area = b.width * b.height
        if b_area > 0 and overlap_area / b_area > threshold:
            return True
    return False


def _attach_captions_to_blocks(page_blocks: List[ContentBlock], page) -> None:
    """Same as original"""
    caption_re = CAPTION_PATTERN
    structured_blocks = [b for b in page_blocks if b.content_type in ('table', 'figure') and b.bbox and not b.caption_label]
    text_blocks = [b for b in page_blocks if b.content_type == "text" and b.bbox]
    used_caption_ids = set()
    
    for sb in structured_blocks:
        sb_rect = fitz.Rect(sb.bbox)
        best_match = None
        
        for tb in text_blocks:
            if id(tb) in used_caption_ids:
                continue
            text = " ".join(tb.content.strip().split())
            match = caption_re.search(text)
            if not match:
                continue
            caption_label = match.group(1) + " " + match.group(2)
            caption_type = 'table' if 'table' in caption_label.lower() else 'figure'
            if caption_type != sb.content_type:
                continue
            tb_rect = fitz.Rect(tb.bbox)
            if tb_rect.y0 >= sb_rect.y1:
                dist = tb_rect.y0 - sb_rect.y1
            else:
                dist = sb_rect.y0 - tb_rect.y1
            if dist < 0 or dist > 150:
                continue
            x_overlap = max(0.0, min(sb_rect.x1, tb_rect.x1) - max(sb_rect.x0, tb_rect.x0))
            if x_overlap < min(sb_rect.width, tb_rect.width) * 0.1:
                continue
            if best_match is None or dist < best_match[0]:
                best_match = (dist, tb, caption_label)
        
        if best_match:
            _, cap_block, caption_label = best_match
            caption_text = cap_block.content.strip()
            if caption_text and caption_text not in sb.content:
                sb.content = f"{caption_text}\n{sb.content}"
            sb.caption_label = caption_label
            used_caption_ids.add(id(cap_block))
    
    page_blocks[:] = [b for b in page_blocks if id(b) not in used_caption_ids]

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
            label = block.caption_label or ""
            output.append(f"[TABLE:{label}]")
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

def extract_pdf_content(data: bytes, use_got_ocr: bool = True) -> str:
    """
    Extract content from PDF with enhanced table/figure detection.
    
    Uses a two-pass approach:
    1. Caption-driven detection: Find "Table N" / "Figure N" in text → locate regions → OCR with Qwen-VL
    2. Fallback detection: Native find_tables() + image block OCR
    
    This ensures we don't miss tables/figures that aren't detected by native methods.
    """
    content_blocks = extract_pdf_content_with_structure(data, use_got_ocr)
    return format_content_blocks_as_text(content_blocks)

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
    print(f"Processing: {uploaded_file.name}")
    print(f"{'='*60}")

    if file_extension == PDF_FILE_EXTENSION:
        content_blocks = None

        # try Docling first (better for structured documents with tables)
        if getattr(Config.Preprocessing, 'USE_DOCLING', True):
            try:
                from docling_loader import extract_pdf_with_docling_from_bytes, format_docling_blocks_as_text
                print(f"🔧 Mode: Docling + Surya OCR (primary)")

                use_ocr = getattr(Config.Preprocessing, 'USE_DOCLING_OCR', True)
                content_blocks = extract_pdf_with_docling_from_bytes(
                    file_bytes,
                    filename=uploaded_file.name,
                    use_ocr=use_ocr
                )
                content = format_docling_blocks_as_text(content_blocks)

                # force fallback if Docling found 0 tables
                tables_found = sum(1 for b in content_blocks if b.content_type == 'table')
                if tables_found > 0:
                    print(f"  ✓ Docling extracted {tables_found} tables successfully")
                else:
                    print(f"  ⚠ Docling returned 0 tables. Running native table pass...")
                    
                    # pass 1: fast native extraction (no OCR)
                    native_blocks = extract_pdf_content_with_structure(file_bytes, use_got_ocr=False)
                    native_tables = [b for b in native_blocks if b.content_type == "table"]
                    
                    if native_tables:
                        print(f"  ✓ Native extractor found {len(native_tables)} tables. Merging with Docling blocks.")
                        content_blocks.extend(native_tables)
                        
                        # sort blocks by page and position
                        def _key(b):
                            y = b.bbox[1] if getattr(b, "bbox", None) else 0.0
                            return (b.page_num, y)
                        content_blocks = sorted(content_blocks, key=_key)
                        
                        # regenerate content with tables
                        content = format_docling_blocks_as_text(content_blocks)
                    else:
                        print(f"Native pass found 0 tables. Escalating to OCR fallback...")
                        # pass 2: Full OCR extraction (expensive but thorough)
                        content_blocks = extract_pdf_content_with_structure(file_bytes, use_got_ocr=True)
                        content = format_content_blocks_as_text(content_blocks)

            except ImportError as e:
                print(f"Docling not available ({e}), falling back to Qwen VL")
                content_blocks = None
            except Exception as e:
                print(f"Docling failed ({e}), falling back to Qwen VL")
                content_blocks = None

        # fallback to Qwen VL OCR if Docling failed or disabled
        if content_blocks is None:
            print(f"🔧 Mode: Native PyMuPDF + Qwen3-VL-4B OCR (fallback)")
            content_blocks = extract_pdf_content_with_structure(file_bytes, use_got_ocr=True)
            content = format_content_blocks_as_text(content_blocks)

        print(f"\nCaching results for future use...")
        # save structured blocks to cache
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

    print(f"Processing complete!")
    print(f"{'='*60}\n")

    return File(name=uploaded_file.name, content=content, content_blocks=content_blocks)

# enhanced with validation and deduplication
def extract_pdf_content_with_structure(data: bytes, use_got_ocr: bool = True) -> List[ContentBlock]:
    """Enhanced extraction with strict table validation"""
    with fitz.open(stream=data, filetype="pdf") as doc:
        all_blocks: List[ContentBlock] = []
        table_extractor = TableExtractor()
        detected_regions: List[tuple] = []
        current_section_header = ""
        tables_by_source = {}
        print(f"Processing {len(doc)} pages with enhanced detection...")
        
        for page_num, page in enumerate(doc):
            print(f"  Page {page_num + 1}/{len(doc)}...", end='\r')
            page_text = page.get_text(sort=True)
            page_blocks: List[ContentBlock] = []
            
            page_headers = extract_section_headers(page_text)
            if page_headers:
                current_section_header = page_headers[-1]
            
            # PASS 1: Caption detection
            caption_regions = find_caption_regions(page, page_text)
            if caption_regions:
                print(f"\n[PASS 1] Page {page_num+1}: Found {len(caption_regions)} captions: {[r['label'] for r in caption_regions]}")
            
            for region in caption_regions:
                content_bbox = expand_region_to_content(page, region['caption_bbox'], region['type'])
                # don't mark as detected until we actually succeed
                added_block = False
                
                if use_got_ocr:
                    try:
                        mat = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=mat, clip=content_bbox)
                        image_bytes = pix.tobytes("png")
                        extracted_text = run_qwen_vl_ocr(image_bytes)
                        if extracted_text and len(extracted_text) > 20:
                            table_result = table_extractor.extract_table_from_ocr(extracted_text)
                            
                            if region['type'] == 'table' and table_result:
                                structured_table, ocr_text = table_result
                                print(f"    [PASS 1] ✓ TABLE: {region['label']}")
                                tables_by_source.setdefault('caption_detection', []).append(region['label'])                               
                                if region['caption'] and region['caption'] not in ocr_text:
                                    ocr_text = f"{region['caption']}\n{ocr_text}"
                                page_blocks.append(ContentBlock(
                                    content=ocr_text,
                                    content_type="table",
                                    page_num=page_num + 1,
                                    bbox=content_bbox,
                                    table_data=structured_table,
                                    caption_label=region['label'],
                                    section_header=current_section_header,
                                ))
                                added_block = True
                            else:
                                print(f"    [PASS 1] Completed - {region['type'].upper()}: {region['label']}")
                                content = f"{region['caption']}\n{extracted_text}" if region['caption'] else extracted_text
                                page_blocks.append(ContentBlock(
                                    content=content,
                                    content_type=region['type'],
                                    page_num=page_num + 1,
                                    bbox=content_bbox,
                                    caption_label=region['label'],
                                    section_header=current_section_header,
                                ))
                                added_block = True
                    except Exception as e:
                        print(f"\nCaption OCR failed for {region['label']} (page {page_num + 1}): {e}")
                # only mark as detected if we successfully added a block
                if added_block:
                    detected_regions.append((page_num, content_bbox))
            
            # PASS 2: Native tables with enhanced validation
            found_tables = page.find_tables()
            if found_tables and len(found_tables.tables) > 0:
                print(f"[PASS 2] Page {page_num+1}: Checking {len(found_tables.tables)} native candidates")
            
            for table in found_tables:
                tbbox = tuple(table.bbox)
                
                # validate table
                is_valid, reason = _is_valid_table(table, page, page_text)
                if not is_valid:
                    print(f"[PASS 2] REJECTED - ({reason})")
                    continue
                if _region_overlaps_any(page_num, tbbox, detected_regions, threshold=0.5):
                    print(f"[PASS 2] SKIPPED - (overlaps)")
                    continue 
                detected_regions.append((page_num, tbbox))
                print(f"[PASS 2] - NATIVE TABLE ADDED")
                tables_by_source.setdefault('native_pymupdf', []).append(f"Page {page_num+1}")
                
                header = table.header.names or [f"Col {i+1}" for i in range(table.col_count)]
                header = [str(h) if h is not None else f"Col {i+1}" for i, h in enumerate(header)]
                
                rows = []
                for row_data in table.extract():
                    if row_data == table.header.names:
                        continue
                    cleaned_row = [str(cell) if cell is not None else "" for cell in row_data]
                    if len(cleaned_row) == len(header):
                        rows.append(dict(zip(header, cleaned_row)))
                
                markdown_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
                for row in rows:
                    markdown_lines.append("| " + " | ".join([str(row.get(h, "")) for h in header]) + " |")
                raw_markdown = "\n".join(markdown_lines)
                
                structured_table = StructuredTable(headers=header, rows=rows, raw_markdown=raw_markdown, num_rows=len(rows), num_cols=len(header))
                page_blocks.append(ContentBlock(content=raw_markdown, content_type="table", page_num=page_num + 1, bbox=tbbox, table_data=structured_table, section_header=current_section_header))
            
            # text and image blocks (same as original - truncated for brevity)
            layout_blocks = page.get_text("dict", sort=True).get("blocks", [])
            for block in layout_blocks:
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                if _region_overlaps_any(page_num, bbox, detected_regions):
                    continue
                
                if block.get("type") == 0:  # text
                    text_content = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_content.append(span.get("text", ""))
                    content = "\n".join([t for t in text_content if t])
                    if content.strip():
                        page_blocks.append(ContentBlock(content=content, content_type="text", page_num=page_num + 1, bbox=bbox, section_header=current_section_header))
            
            def _sort_key(b: ContentBlock):
                if b.bbox:
                    x0, y0, x1, y1 = b.bbox
                    return (y0, x0)
                return (float("inf"), float("inf"))
            
            page_blocks.sort(key=_sort_key)
            _attach_captions_to_blocks(page_blocks, page)
            all_blocks.extend(page_blocks)
        
        # deduplication
        all_blocks = _merge_duplicate_tables(all_blocks)
        
        # DIAGNOSTIC REPORT
        tables = sum(1 for b in all_blocks if b.content_type == "table")
        figures = sum(1 for b in all_blocks if b.content_type == "figure")
        print(f"\nExtraction complete: {tables} tables, {figures} figures, {len(all_blocks)} total blocks")
        
        print(f"\nTABLE EXTRACTION REPORT:")
        for source, labels in tables_by_source.items():
            print(f"  {source}: {len(labels)} tables - {labels[:5]}{'...' if len(labels) > 5 else ''}")
        
        return all_blocks

def cleanup_ocr_model():
    """Free GPU memory after processing (Qwen VL + Docling/Surya)."""
    import gc
    
    print("\n" + "="*50)
    print("🧹 Cleaning up OCR models to free GPU memory...")
    print("="*50)
    
    # unload Qwen VL
    unload_qwen_vl_model()
    
    # try to unload Docling/Surya models
    try:
        from docling_loader import unload_docling_models
        unload_docling_models()
    except ImportError:
        pass
    
    # force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  Final VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("GPU memory freed!")
    print("="*50 + "\n")