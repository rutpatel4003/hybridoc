"""
Docling-based PDF loader with Surya OCR support.
Primary parser for structured documents (academic papers, reports, etc.)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any
import re
import io
import os
import sys

if sys.platform == "win32":
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# docling imports with error handling
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    print(f"Warning: Docling not available: {e}")

# try to import TableFormerMode and TableStructureOptions
try:
    from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions
    TABLE_FORMER_AVAILABLE = True
except ImportError:
    try:
        # try older import path
        from docling.datamodel.pipeline_options import TableFormerMode
        from docling.datamodel.pipeline_options import TableStructureOptions
        TABLE_FORMER_AVAILABLE = True
    except ImportError:
        TABLE_FORMER_AVAILABLE = False

# try to import docling_core types
try:
    from docling_core.types.doc import PictureItem
    DOCLING_CORE_AVAILABLE = True
except ImportError:
    DOCLING_CORE_AVAILABLE = False

from table_intelligence import StructuredTable
from config import Config

# global reference to track the Docling converter for cleanup
_GLOBAL_CONVERTER = None
def _is_table_like(item) -> bool:
    """
    Capability-based table detection (robust across Docling versions).
    
    Treats as table if:
    - Type name contains "table" OR
    - item.label == "table" OR  
    - Has export method like export_to_dataframe()
    """
    tname = type(item).__name__.lower()
    label = str(getattr(item, "label", "")).lower()
    
    if "table" in tname or label == "table":
        return True
    
    # capability-based check
    for method in ("export_to_dataframe", "export_to_df", "to_pandas", "to_dataframe"):
        if hasattr(item, method):
            return True
    
    return False

@dataclass
class ContentBlock:
    """Represents a block of content from PDF"""
    content: str
    content_type: str  # 'text', 'table', 'figure'
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

def _get_docling_converter(use_ocr: bool = True) -> "DocumentConverter":
    """
    Create a Docling converter with optimal settings for table extraction.
    Uses Surya OCR for scanned documents and table images.
    """
    global _GLOBAL_CONVERTER
    
    # return cached converter if available
    if _GLOBAL_CONVERTER is not None:
        print("  Reusing existing Docling converter (singleton)")
        return _GLOBAL_CONVERTER
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling is not installed. Run: pip install docling docling-surya")

    print("Creating NEW Docling converter (first time)...")
    
    # build pipeline options
    pipeline_kwargs = {
        "do_ocr": use_ocr,
        "do_table_structure": True,
        "generate_picture_images": True,  # enable image extraction for Qwen VL OCR
        "generate_page_images": True,  # keep page images for figure extraction
        "images_scale": 2.0,  # higher resolution for better OCR
        "table_structure_options": TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True  # improve cell detection
        ) if TABLE_FORMER_AVAILABLE else None,
    }

    # add table structure mode if available
    if TABLE_FORMER_AVAILABLE:
        try:
            pipeline_kwargs["table_structure_options"] = TableStructureOptions(mode=TableFormerMode.ACCURATE)
        except Exception:
            pass  # skip if API is different

    pipeline_options = PdfPipelineOptions(**pipeline_kwargs)
    # try to use Surya OCR if available
    if use_ocr:
        try:
            from docling_surya import SuryaOcrOptions
            pipeline_options.ocr_options = SuryaOcrOptions(lang=["en"])
            pipeline_options.allow_external_plugins = True
            print("Using Surya OCR backend")
        except ImportError:
            print("  Surya OCR not available, using default OCR")
        except Exception as e:
            print(f"  Could not configure Surya OCR: {e}")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    # store global reference for cleanup
    _GLOBAL_CONVERTER = converter
    return converter

def unload_docling_models():
    """Unload Docling/Surya models to free GPU memory."""
    global _GLOBAL_CONVERTER
    import gc
    import torch

    print("🧹 Unloading Docling/Surya models...")
    
    # delete the global converter instance
    if _GLOBAL_CONVERTER is not None:
        print("  Deleting Docling converter...")
        try:
            # try to explicitly cleanup any model attributes
            if hasattr(_GLOBAL_CONVERTER, 'model'):
                del _GLOBAL_CONVERTER.model
            if hasattr(_GLOBAL_CONVERTER, '_model'):
                del _GLOBAL_CONVERTER._model
        except:
            pass
        _GLOBAL_CONVERTER = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  VRAM after converter delete: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    models_cleared = 0
    try:
        from surya.model.detection import segformer
        for attr in ['model', 'processor', '_model', '_processor']:
            if hasattr(segformer, attr):
                obj = getattr(segformer, attr, None)
                if obj is not None:
                    delattr(segformer, attr)
                    models_cleared += 1
    except Exception as e:
        pass  # module may not be loaded
    # clear recognition models  
    try:
        from surya.model import recognition
        for attr in ['model', 'processor', '_model', '_processor']:
            if hasattr(recognition, attr):
                obj = getattr(recognition, attr, None)
                if obj is not None:
                    delattr(recognition, attr)
                    models_cleared += 1
    except Exception as e:
        pass
    # clear any cached models in surya submodules
    try:
        import sys
        surya_modules = [m for m in sys.modules if m.startswith('surya')]
        for mod_name in surya_modules:
            mod = sys.modules.get(mod_name)
            if mod:
                # look for any model-like attributes
                for attr in list(dir(mod)):
                    if attr.startswith('_'):
                        continue
                    try:
                        obj = getattr(mod, attr, None)
                        # check if it's a PyTorch model or has CUDA tensors
                        if obj is not None:
                            if hasattr(obj, 'to') and hasattr(obj, 'parameters'):
                                delattr(mod, attr)
                                models_cleared += 1
                            elif hasattr(obj, 'device') and 'cuda' in str(getattr(obj, 'device', '')):
                                delattr(mod, attr)
                                models_cleared += 1
                    except:
                        pass
    except:
        pass
    # clear Docling's EasyOCR if used as fallback
    try:
        import easyocr
        if hasattr(easyocr, 'Reader'):
            # EasyOCR caches readers
            pass
    except:
        pass

    print(f"  Cleared {models_cleared} cached model references")
    # multiple rounds of garbage collection
    for _ in range(3):
        gc.collect()

    # clear CUDA cache aggressively
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # force reset of CUDA memory stats
        torch.cuda.reset_peak_memory_stats()
        print(f"  VRAM after Docling cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def cleanup_all_models():
    """Cleanup all OCR models (Docling, Surya, Qwen VL) to free GPU memory."""
    import gc
    import torch
    from pdf_loader import unload_qwen_vl_model
    print("\n" + "="*50)
    print("🧹 Cleaning up OCR models to free GPU memory...")
    print("="*50)
    unload_qwen_vl_model()
    unload_docling_models()
    # final aggressive cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # check final memory state
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  Final VRAM usage: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        # if still high, try to reset
        if allocated > 1.0:
            print("Memory still high, forcing additional cleanup...")
            gc.collect()
            torch.cuda.empty_cache()

    print("GPU memory freed!")
    print("="*50 + "\n")

def _table_to_structured(table_item, doc=None) -> Optional[StructuredTable]:
    """Convert Docling TableItem to our StructuredTable format."""
    try:
        # get table as pandas DataFrame
        if doc is not None:
            df = table_item.export_to_dataframe(doc)
        else:
            df = table_item.export_to_dataframe()
        if df.empty:
            return None

        headers = [str(col) for col in df.columns.tolist()]
        rows = []

        for _, row in df.iterrows():
            row_dict = {headers[i]: str(val) if val is not None else ""
                       for i, val in enumerate(row)}
            rows.append(row_dict)
        # generate markdown
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            row_values = [str(row.get(h, "")) for h in headers]
            md_lines.append("| " + " | ".join(row_values) + " |")

        raw_markdown = "\n".join(md_lines)
        return StructuredTable(
            headers=headers,
            rows=rows,
            raw_markdown=raw_markdown,
            num_rows=len(rows),
            num_cols=len(headers),
        )
    except Exception as e:
        print(f"  Warning: Failed to convert table: {e}")
        return None


def extract_pdf_with_docling(file_path: str, use_ocr: bool = True) -> List[ContentBlock]:
    """
    Extract content from PDF using Docling with Surya OCR.
    """
    print(f"Docling: Processing {Path(file_path).name}...")

    converter = _get_docling_converter(use_ocr)
    result = converter.convert(file_path)
    doc = result.document

    content_blocks: List[ContentBlock] = []
    current_section = ""

    # track tables and figures separately for proper labeling
    table_count = 0
    figure_count = 0

    # iterate through document elements
    for element in doc.iterate_items():
        item = element[0] if isinstance(element, tuple) else element
        # get page number
        page_num = 1
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'page_no'):
                    page_num = prov.page_no
                    break
        # get bounding box
        bbox = None
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'bbox'):
                    b = prov.bbox
                    bbox = (b.l, b.t, b.r, b.b) if hasattr(b, 'l') else None
                    break
        # handle different item types - use capability-based detection
        item_type = type(item).__name__
        if _is_table_like(item):
            table_count += 1
            structured_table = _table_to_structured(item, doc)
            # get caption - try multiple attribute names for compatibility
            caption = None
            for attr in ("caption_text", "caption", "title"):
                if not hasattr(item, attr):
                    continue

                value = getattr(item, attr, None)

                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        try:
                            value = value(doc)
                        except Exception:
                            value = None
                    except Exception:
                        value = None

                if value:
                    s = str(value).strip()

                    if s.startswith("<bound method"):
                        continue

                    caption = f"Table {table_count}: {s}"
                    break

            if not caption:
                caption = f"Table {table_count}"

            if structured_table:
                content = f"{caption}\n{structured_table.raw_markdown}"
                content_blocks.append(ContentBlock(
                    content=content,
                    content_type="table",
                    page_num=page_num,
                    bbox=bbox,
                    table_data=structured_table,
                    caption_label=f"Table {table_count}",
                    section_header=current_section,
                ))
                print(f"    Page {page_num}: Extracted Table {table_count} ({structured_table.num_rows} rows, {structured_table.num_cols} cols)")
            else:
                # fallback: export as markdown text
                try:
                    md_content = item.export_to_markdown()
                    content_blocks.append(ContentBlock(
                        content=f"{caption}\n{md_content}",
                        content_type="table",
                        page_num=page_num,
                        bbox=bbox,
                        caption_label=f"Table {table_count}",
                        section_header=current_section,
                    ))
                except Exception as e:
                    # don't drop tables silently - create placeholder
                    print(f"Table {table_count} export failed: {e}")
                    content_blocks.append(ContentBlock(
                        content=f"{caption}\n[Docling table extraction failed: {e}]",
                        content_type="table",
                        page_num=page_num,
                        bbox=bbox,
                        caption_label=f"Table {table_count}",
                        section_header=current_section,
                    ))

                except:
                    pass

        elif item_type == 'PictureItem' or (DOCLING_CORE_AVAILABLE and hasattr(item, '__class__') and item.__class__.__name__ == 'PictureItem'):
            figure_count += 1
            caption = f"Figure {figure_count}"
            if hasattr(item, "caption_text"):
                value = getattr(item, "caption_text", None)
                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        try:
                            value = value(doc)
                        except Exception:
                            value = None
                    except Exception:
                        value = None

                if value:
                    s = str(value).strip()
                    if not s.startswith("<bound method"):
                        caption = f"Figure {figure_count}: {s}"
            # fet any text/description associated with the figure
            content = caption
            if hasattr(item, 'text') and item.text:
                content = f"{caption}\n{item.text}"

            # try to OCR the figure image using Qwen VL for better content extraction
            figure_ocr_text = ""
            try:
                # try to get the image from PictureItem (multiple API patterns)
                pil_image = None

                # method 1: Direct image attribute 
                if hasattr(item, 'image') and item.image is not None:
                    if hasattr(item.image, 'pil_image') and item.image.pil_image is not None:
                        pil_image = item.image.pil_image

                # method 2: get_image method with document
                if pil_image is None and hasattr(item, 'get_image'):
                    try:
                        pil_image = item.get_image(doc)
                    except:
                        pass

                # method 3: Check for image data in prov
                if pil_image is None and hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'image') and prov.image is not None:
                            if hasattr(prov.image, 'pil_image'):
                                pil_image = prov.image.pil_image
                                break

                if pil_image is not None:
                    # convert PIL image to bytes for Qwen VL
                    from pdf_loader import run_qwen_vl_ocr
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    figure_ocr_text = run_qwen_vl_ocr(img_bytes)
                    if figure_ocr_text and len(figure_ocr_text) > 20:
                        content = f"{caption}\n{figure_ocr_text}"
                        print(f"    Page {page_num}: Extracted Figure {figure_count} (Qwen VL OCR: {len(figure_ocr_text)} chars)")
                    else:
                        print(f"    Page {page_num}: Extracted Figure {figure_count} (caption only)")
                else:
                    print(f"    Page {page_num}: Extracted Figure {figure_count} (no image data - check generate_picture_images)")
            except Exception as e:
                import traceback
                print(f"    Page {page_num}: Extracted Figure {figure_count} (OCR failed: {e})")
                traceback.print_exc()
            content_blocks.append(ContentBlock(
                content=content,
                content_type="figure",
                page_num=page_num,
                bbox=bbox,
                caption_label=f"Figure {figure_count}",
                section_header=current_section,
            ))

        else:
            # text content
            text = ""
            if hasattr(item, 'text'):
                text = item.text
            elif hasattr(item, 'export_to_markdown'):
                text = item.export_to_markdown()
            if text and text.strip():
                # check for section headers
                if hasattr(item, 'label') and item.label in ['section_header', 'title']:
                    current_section = text.strip()
                content_blocks.append(ContentBlock(
                    content=text.strip(),
                    content_type="text",
                    page_num=page_num,
                    bbox=bbox,
                    section_header=current_section,
                ))
    # count block types for verification
    from collections import Counter
    block_types = Counter(b.content_type for b in content_blocks)
    text_count = block_types.get('text', 0) + block_types.get('paragraph', 0) + block_types.get('section_header', 0)

    print(f"Docling: Extracted {table_count} tables, {figure_count} figures, {len(content_blocks)} total blocks")
    print(f"--> Block breakdown: {text_count} text, {table_count} tables, {figure_count} figures")

    # warn if suspiciously low text extraction
    if text_count < 10 and len(content_blocks) > 20:
        print(f"WARNING: Only {text_count} text blocks for {len(content_blocks)} total blocks - text extraction may have failed!")

    return content_blocks


def extract_pdf_with_docling_from_bytes(data: bytes, filename: str = "document.pdf", use_ocr: bool = True) -> List[ContentBlock]:
    """
    Extract content from PDF bytes using Docling.
    Writes to temp file since Docling works with file paths.
    """
    import tempfile
    import os

    # write bytes to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        blocks = extract_pdf_with_docling(tmp_path, use_ocr)
        return blocks
    finally:
        # cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

def format_docling_blocks_as_text(content_blocks: List[ContentBlock]) -> str:
    """
    Convert content blocks to text format compatible with data_ingestor.py.
    OPTIMIZED: Uses only searchable row-indexed text for tables to save context.
    """
    output: List[str] = []
    current_page = 0

    for block in content_blocks:
        # add page marker when page changes
        if block.page_num != current_page:
            output.append(f"\n--- PAGE {block.page_num} ---\n")
            current_page = block.page_num

        if block.content_type == "table":
            # 1. robust Labeling
            label = block.caption_label or ""
            if not label:
                m = re.match(r'^\s*(Table\s+\d+(?:\.\d+)?)', (block.content or ""), re.IGNORECASE)
                label = m.group(1).title() if m else "Data Table"
            
            output.append(f"[TABLE:{label}]")

            # 2. use only searchable text if available
            if block.table_data:
                # this provides the dense 'Row X: Header=Value' format
                output.append(block.table_data.to_searchable_text())
            elif block.content and block.content.strip():
                # fallback to raw content only if structured table_data is missing
                output.append(block.content.strip())

            output.append("[/TABLE]")
            output.append("")

        elif block.content_type == "figure":
            output.append("[FIGURE]")
            output.append(block.content.strip())
            output.append("[/FIGURE]")
            output.append("")

        else:
            # standard text
            output.append(block.content.strip())
            output.append("")

    return "\n".join(output)