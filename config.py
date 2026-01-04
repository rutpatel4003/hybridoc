import os
from pathlib import Path
from loguru import logger
import sys
class Config:
    SEED=42
    ALLOWED_FILE_EXTENSIONS = set(['.pdf', '.md', '.txt'])

    class Model:
            NAME = 'qwen3:4b-instruct'
            TEMPERATURE = 0.6

    class Preprocessing: 
        CHUNK_SIZE = 1024
        CHUNK_OVERLAP = 128
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        RERANKER = 'ms-marco-MiniLM-L-12-v2'
        LLM = 'qwen3:4b-instruct'
        CONTEXTUALIZE_CHUNKS = False
        N_SEMANTIC_RESULTS = 6
        N_BM25_RESULTS = 6

    class Chatbot: 
        N_CONTEXT_RESULTS = 4
        GRADING_MODE = False

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

    
        


