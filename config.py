import os
from pathlib import Path
from loguru import logger
import sys
class Config:
    SEED=42
    ALLOWED_FILE_EXTENSIONS = set(['.pdf', '.md', '.txt'])

    class Model:
            NAME = 'qwen3:4b'
            TEMPERATURE = 0.6
            HISTORY_REPHRASER_LLM_NAME = 'qwen3:0.6b'

    class Preprocessing: 
        CHUNK_SIZE = 512
        CHUNK_OVERLAP = 100
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        RERANKER = 'ms-marco-MiniLM-L-12-v2'
        LLM = 'qwen3:4b'
        CONTEXTUALIZE_CHUNKS = False
        N_SEMANTIC_RESULTS = 10
        N_BM25_RESULTS = 10

    class Chatbot: 
        N_CONTEXT_RESULTS = 5

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME/"data"
        


