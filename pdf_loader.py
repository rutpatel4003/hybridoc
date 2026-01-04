from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF
from streamlit.runtime.uploaded_file_manager import UploadedFile
from config import Config
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import io
import hashlib

TEXT_FILE_EXTENSION = ".txt"
MD_FILE_EXTENSION = '.md'
PDF_FILE_EXTENSION = ".pdf"

# global variables for lazy loading
_got_model = None
_got_processor = None

def get_got_ocr_model():
    """
    Lazy-load GOT-OCR2 model.
    """
    global _got_model, _got_processor
    
    if _got_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # load model normally with float16 for GPU efficiency
        _got_model = AutoModelForImageTextToText.from_pretrained(
            "stepfun-ai/GOT-OCR-2.0-hf",
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        
        _got_processor = AutoProcessor.from_pretrained(
            "stepfun-ai/GOT-OCR-2.0-hf",
            use_fast=True
        )
        
        # set to eval mode
        _got_model.eval()        
        if device == "cuda":
            print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    return _got_model, _got_processor

def run_got_ocr_on_image(image_bytes: bytes, format_output: bool = True) -> str:
    """
    Run GOT-OCR2 on image bytes and return extracted text.
    """
    try:
        model, processor = get_got_ocr_model()
        
        # convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # validate image
        if image.size[0] < 10 or image.size[1] < 10:
            return ""  # skip tiny images
        
        # process image with appropriate settings
        inputs = processor(
            image,
            return_tensors="pt",
            format=format_output  # enable for lateX, tables, formulas
        )
        
        # move to GPU if available
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # generate with memory-efficient settings
        with torch.no_grad(): 
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=2048,
                num_beams=1,  # greedy decoding
            )
        
        # decode output
        output_text = processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # clear cuda cache after each inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text.strip()
        
    except Exception as e:
        import traceback
        print(f"GOT-OCR2 failed: {e}")
        print(traceback.format_exc())
        return ""

def extract_pdf_content(data: bytes, use_got_ocr: bool = True) -> str:
    """
    Extract content from PDF with GOT-OCR2 for images.
    """
    with fitz.open(stream=data, filetype="pdf") as doc:
        full_text = []
        
        for page_num, page in enumerate(doc):
            print(f"Processing page {page_num + 1}/{len(doc)}...")
            
            # extract text blocks
            blocks = page.get_text("dict", sort=True)["blocks"]
            
            for block in blocks:
                # type 0 = text block
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            full_text.append(span["text"])
                    full_text.append("\n")
                
                # type 1 = image block
                elif block["type"] == 1 and use_got_ocr:
                    try:
                        # extract image from the page region using bbox
                        bbox = block.get("bbox")
                        if bbox:
                            # extract image with 2x zoom for better quality
                            mat = fitz.Matrix(2, 2)
                            pix = page.get_pixmap(matrix=mat, clip=bbox)
                            image_bytes = pix.tobytes("png")
                            
                            # run GOT-OCR2
                            extracted_text = run_got_ocr_on_image(
                                image_bytes,
                                format_output=True
                            )
                            
                            if extracted_text:
                                full_text.append(f"\n[IMAGE CONTENT]\n{extracted_text}\n[/IMAGE CONTENT]\n")
                        
                    except Exception as e:
                        print(f"Failed to process image on page {page_num + 1}: {e}")
            
            full_text.append("\nPAGE BREAK\n")
    return "\n".join(full_text)

@dataclass 
class File:
    name: str
    content: str

def load_uploaded_file(uploaded_file: UploadedFile) -> File:
    """
    load and process uploaded file with persistent disk caching.
    """
    file_extension = Path(uploaded_file.name).suffix
    
    if file_extension not in Config.ALLOWED_FILE_EXTENSIONS:
        raise ValueError(
            f"Invalid file extension: {file_extension} for file {uploaded_file.name}"
        )
    
    # get raw bytes to generate a unique hash for this specific file content
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    # define cache directory and file path
    cache_dir = Config.Path.DATA_DIR / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{file_hash}.txt"

    # load from disk if file is processed already
    if cache_path.exists():
        print(f"🔄 Cache hit: Loading processed text for {uploaded_file.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            content = f.read()
        return File(name=uploaded_file.name, content=content)
    
    # if not in cache, run the ocr extraction
    print(f"Running extraction for {uploaded_file.name}")
    if file_extension == PDF_FILE_EXTENSION:
        content = extract_pdf_content(file_bytes, use_got_ocr=True)
    else:
        content = file_bytes.decode("utf-8")
    
    # save the result to avoid running OCR on this file again
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    return File(name=uploaded_file.name, content=content)

# cleanup function to free GPU memory when done
def cleanup_got_model():
    """call this when you're done processing to free GPU memory."""
    global _got_model, _got_processor
    
    if _got_model is not None:
        del _got_model
        del _got_processor
        _got_model = None
        _got_processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GOT-OCR2 model unloaded, GPU memory freed")