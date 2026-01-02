from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from streamlit.runtime.uploaded_file_manager import UploadedFile
from config import Config
import io

TEXT_FILE_EXTENSION = ".txt"
MD_FILE_EXTENSION = '.md'
PDF_FILE_EXTENSION = ".pdf"

# initialize OCR engine once (Global load for speed)
# use det_use_gpu=True if you have CUDA configured properly, else False is still fast
try:
    ocr_engine = RapidOCR(det_use_gpu=True)
except:
    ocr_engine = RapidOCR(det_use_gpu=False)

@dataclass 
class File:
    name: str
    content: str

def extract_pdf_content(data: bytes) -> str:
    """
    Iterates through the PDF block by block.
    - If text: keeps it.
    - If image: runs OCR and injects the text in-place.
    """
    # open PDF from bytes
    with fitz.open(stream=data, filetype="pdf") as doc:
        full_text = []
        
        for page_num, page in enumerate(doc):
            # get_text("dict") allows us to separate images from text blocks
            blocks = page.get_text("dict", sort=True)["blocks"]
            
            for block in blocks:
                # type 0 = Text
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            full_text.append(span["text"])
                    full_text.append("\n")
                
                # type 1 = Image
                elif block["type"] == 1:
                    try:
                        image_bytes = block["image"]
                        # run OCR on the image bytes
                        ocr_result, _ = ocr_engine(image_bytes)
                        
                        if ocr_result:
                            # extract just the text parts
                            extracted_text = " ".join([res[1] for res in ocr_result])
                            if extracted_text.strip():
                                full_text.append(f"\n[IMAGE CONTENT: {extracted_text}]\n")
                    except Exception as e:
                        # fail silently on bad images to keep the pipeline moving
                        pass
                        
            full_text.append("\n--- PAGE BREAK ---\n")
            
    return "\n".join(full_text)

def load_uploaded_file(uploaded_file: UploadedFile) -> File:
    file_extension = Path(uploaded_file.name).suffix
    if file_extension not in Config.ALLOWED_FILE_EXTENSIONS:
        raise ValueError(f"Invalid file extension: {file_extension} for file {uploaded_file.name}")
    
    if file_extension == PDF_FILE_EXTENSION:
        return File(name=uploaded_file.name, content=extract_pdf_content(uploaded_file.getvalue()))
    
    # text/MD fallback
    return File(name=uploaded_file.name, content=uploaded_file.getvalue().decode("utf-8"))