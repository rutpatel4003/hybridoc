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
            NAME = 'qwen3:4b-instruct'
            TEMPERATURE = 0.1

    class Preprocessing:
        CHUNK_SIZE = 1024
        CHUNK_OVERLAP = 150
        EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
        RERANKER = 'cross-encoder/ms-marco-MiniLM-L12-v2'
        CONTEXTUALIZE_CHUNKS = False
        N_SEMANTIC_RESULTS = 6
        N_BM25_RESULTS = 6
        ENABLE_PARENT_CHILD = True
        PARENT_CHUNK_SIZE = 2048  # larger chunks for LLM context
        CHILD_CHUNK_SIZE = 256    # smaller chunks for precise retrieval
        USE_RRF = True  # use Reciprocal Rank Fusion 
        RRF_K = 60 

    class Chatbot:
        N_CONTEXT_RESULTS = 3
        GRADING_MODE = False
        ENABLE_QUERY_ROUTER = True
        ROUTER_HISTORY_WINDOW = 4
        ENABLE_HYDE = False  
        ENABLE_MULTI_QUERY = True   

    class Eval:
        """Evaluation settings"""
        GOLD_SET_PATH = "eval/gold_set.json"
        K = 4  # recall@k
        OUTPUT_DIR = "eval/results"

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME/"data"
        VECTOR_DB_DIR = DATA_DIR / 'chroma_db'

    class Performance: 
         """
         Production performance settings
         """
         EMBEDDING_BATCH_SIZE = 32
         USE_MULTIPROCESSING = True
         MAX_WORKERS = 4
         ENABLE_QUERY_CACHE = True
         CACHE_TTL_SECONDS = 3600
         CLEAR_GPU_AFTER_INDEXING = True

    
        


