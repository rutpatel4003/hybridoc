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
        EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
        RERANKER = 'cross-encoder/ms-marco-MiniLM-L12-v2'
        CONTEXTUALIZE_CHUNKS = False
        N_SEMANTIC_RESULTS = 6
        N_BM25_RESULTS = 6
        ENABLE_PARENT_CHILD = False
        PARENT_CHUNK_SIZE = 3072  # larger chunks for LLM context
        CHILD_CHUNK_SIZE = 500    # smaller chunks for precise retrieval
        USE_RRF = True  # use Reciprocal Rank Fusion
        RRF_K = 60
        USE_PYMUPDF4LLM = False  # disabled - use native fitz extraction with Qwen3-VL OCR
        ENABLE_IMAGE_TABLE_CAPTION_OCR = True  # Qwen3-VL-4B for table/figure OCR
        IMAGE_TABLE_CAPTION_HEIGHT = 140
        ENABLE_TABLE_SEMANTIC_ENRICHMENT = True  # add semantic descriptions to tables for better retrieval
        ENABLE_METADATA_EXTRACTION = True  # extract rich metadata (document + chunk level)

    class Chatbot:
        N_CONTEXT_RESULTS = 3
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
         EMBEDDING_BATCH_SIZE = 32
         USE_MULTIPROCESSING = True
         MAX_WORKERS = 6
         ENABLE_QUERY_CACHE = True
         CACHE_TTL_SECONDS = 3600
         CLEAR_GPU_AFTER_INDEXING = True
         CACHE_SIMILARITY_THRESHOLD = 0.90
         CACHE_MAX_SIZE = 1000
         CACHE_EVICT_EVERY = 100

    
        


