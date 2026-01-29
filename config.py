import os
from pathlib import Path
from loguru import logger
import sys

def get_device() -> str:
    """
    Auto-detect best available device for ML models.
    Returns 'cuda' if GPU available, else 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'

class Config:
    SEED = 42
    ALLOWED_FILE_EXTENSIONS = set(['.pdf', '.md', '.txt'])
    DEVICE = get_device()

    class Model:
            NAME = 'qwen3:4b-instruct'  # fast for user-facing responses
            TEMPERATURE = 0.1

    class Preprocessing:
        CHUNK_SIZE = 2000
        CHUNK_OVERLAP = 300

        # cross-page stitching: helps when a sentence continues on the next page (even if tables/figures are in between)
        ENABLE_CROSS_PAGE_BRIDGING = True
        CROSS_PAGE_CONTEXT_CHARS = 800 # from 600
        CROSS_PAGE_MIN_TEXT_CHARS = 120
        EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
        EMBEDDING_DEVICE = "auto"  # "auto", "cuda", or "cpu" - use "cpu" if GPU OOM after OCR
        RERANKER = 'cross-encoder/ms-marco-MiniLM-L12-v2'
        CONTEXTUALIZE_CHUNKS = False
        N_SEMANTIC_RESULTS = 10
        N_BM25_RESULTS = 10
        ENABLE_PARENT_CHILD = False
        PARENT_CHUNK_SIZE = 3072  # larger chunks for LLM context
        CHILD_CHUNK_SIZE = 500    # smaller chunks for precise retrieval
        USE_RRF = True  # use Reciprocal Rank Fusion
        RRF_K = 60
        
        # PDF Parsing options
        USE_DOCLING = True  # use Docling + Surya OCR as primary parser (best for tables)
        USE_DOCLING_OCR = True  # enable Surya OCR in Docling for scanned documents
        ENABLE_IMAGE_TABLE_CAPTION_OCR = True  # Qwen3-VL-4B for table/figure OCR (fallback)
        IMAGE_TABLE_CAPTION_HEIGHT = 140
        ENABLE_TABLE_SEMANTIC_ENRICHMENT = True  # add semantic descriptions to tables
        ENABLE_METADATA_EXTRACTION = True  # extract rich metadata
        
        # increased region sizes for better table capture
        TABLE_FIGURE_REGION_HEIGHT = 500  # pixels to expand for table/figure content (increased from 450)
        TABLE_FIGURE_CONTEXT_CHARS = 600  # context chars before/after tables/figures (increased from 500)
        
        # PyMuPDF fallback control
        ENABLE_PYMUPDF_FALLBACK = True  # use native PyMuPDF tables even when Docling succeeds

    class Chatbot:
        N_CONTEXT_RESULTS = 7
        GRADING_MODE = False
        ENABLE_QUERY_ROUTER = True
        ROUTER_HISTORY_WINDOW = 4
        ENABLE_HYDE = False
        ENABLE_MULTI_QUERY = False
        ENABLE_CONTEXTUAL_COMPRESSION = False   
        ENABLE_QUERY_DECOMPOSITION = True
        DECOMPOSE_MAX_SUBQUESTIONS = 3
        DECOMPOSE_MIN_WORDS = 10
        ENABLE_QUERY_SCORING = True
        QUERY_SCORE_TOP_K = 3
        QUERY_SCORE_WEIGHTS = {"semantic": 0.7, "lexical": 0.3}
        QUERY_SCORE_THRESHOLDS = {"high": 0.75, "medium": 0.55}
        COMPRESSION_MIN_RATIO = 0.8
        COMPRESSION_MAX_DOC_LENGTH = 1200
        DEBUG_LLM_CONTEXT = True
        DEBUG_CONTEXT_FILEPATH = "llm_context_debug.txt"

        # Neighbor Expansion: expand retrieved chunks to include adjacent chunks
        # solves: section headers at page end, content split by tables/figures, mid-sentence breaks
        ENABLE_NEIGHBOR_EXPANSION = True
        NEIGHBOR_MAX_PER_DIRECTION = 1  # max neighbors to add per direction (forward/backward)
        NEIGHBOR_MIN_OVERLAP_RATIO = 0.15  # minimum content overlap to trigger expansion (15%)
        NEIGHBOR_ENABLE_FORWARD = True  # expand to next chunks
        NEIGHBOR_ENABLE_BACKWARD = True  # expand to previous chunks
        NEIGHBOR_DEBUG = False  # print expansion debug info

    class Eval:
        """Evaluation settings"""
        GOLD_SET_PATH = "eval/gold_set.json"
        K = 4  # recall@k
        OUTPUT_DIR = "eval/results"
        ENABLE_LLM_JUDGE = True
        JUDGE_MODEL = "qwen3:4b-thinking"  # Model for LLM-as-judge evaluation
        JUDGE_TEMPERATURE = 0
        FAITHFULNESS_THRESHOLD = 0.7
        HALLUCINATION_THRESHOLD = 0.4 

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME/"data"
        VECTOR_DB_DIR = DATA_DIR / 'qdrant_db_1'

    class Performance: 
         """
         Production performance settings
         """
         EMBEDDING_BATCH_SIZE = 8  # batch size for embedding generation (conservative for 6GB GPU)
         USE_MULTIPROCESSING = True
         MAX_WORKERS = 6
         ENABLE_QUERY_CACHE = True
         CACHE_TTL_SECONDS = 3600
         CLEAR_GPU_AFTER_INDEXING = True
         CACHE_SIMILARITY_THRESHOLD = 0.97
         CACHE_MAX_SIZE = 1000
         CACHE_EVICT_EVERY = 100

    class Tables:
        """
        General-purpose table support:
        ensure tables remain retrievable without domain-specific keyword triggers.
        """
        ENABLE_STRATIFIED_TABLE_RETRIEVAL = True

        # candidate counts for the table-only retriever
        N_TABLE_SEMANTIC_RESULTS = 6
        N_TABLE_BM25_RESULTS = 6

        # how strongly table-only results influence fused ranking
        TABLE_LIST_WEIGHT = 1.35

        # reranker guardrail (Priority 1 uses this)
        MIN_TABLES_IN_CONTEXT = 2