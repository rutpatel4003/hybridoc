import os
from pathlib import Path

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
            # llama.cpp GGUF model path (from unsloth/Qwen3-4B-Thinking-2507-GGUF)
            GGUF_PATH = Path(__file__).parent / "models" / "Qwen3-4B-Instruct-2507-UD-Q5_K_XL.gguf"
            TEMPERATURE = 0.1
            N_CTX = 18432 # increased context window (thinking tokens stripped before context build)
            N_GPU_LAYERS = -1  # -1 = full GPU offload, 0 = CPU-only
            N_BATCH = 1024  # prompt processing batch size (512 = good balance)
            N_THREADS = 8  # CPU threads (if not using GPU)
            MAX_OUTPUT_TOKENS = 1024  # limit output length

    class Preprocessing:
        CHUNK_SIZE = 2000
        CHUNK_OVERLAP = 300

        # cross-page stitching: helps when a sentence continues on the next page (even if tables/figures are in between)
        ENABLE_CROSS_PAGE_BRIDGING = True
        CROSS_PAGE_CONTEXT_CHARS = 800 # from 600
        CROSS_PAGE_MIN_TEXT_CHARS = 120

        # contextual retrieval: add 1-2 sentence context to each chunk
        ENABLE_CONTEXTUAL_RETRIEVAL = True
        CONTEXTUAL_RETRIEVAL_MAX_CHUNKS = 500  # reduced for speed if re-enabled
        # embedding model: google's embeddinggemma-300m
        # requires task prefixes: "search_query: " and "search_document: "
        EMBEDDING_MODEL = "google/embeddinggemma-300m"
        EMBEDDING_DEVICE = "cuda"  # "auto", "cuda", or "cpu" - use "cpu" if GPU OOM after OCR
        CONTEXTUALIZE_CHUNKS = False
        N_SEMANTIC_RESULTS = 20
        N_BM25_RESULTS = 20
        ENABLE_PARENT_CHILD = False
        PARENT_CHUNK_SIZE = 3072  # larger chunks for LLM context
        CHILD_CHUNK_SIZE = 384    # smaller chunks for precise retrieval
        USE_RRF = True  # use Reciprocal Rank Fusion
        RRF_K = 30
        
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

        # RAPTOR Lite: Hierarchical summarization
        ENABLE_RAPTOR_LITE = False  # create cluster summaries for better multi-hop retrieval
    class Chatbot:
        N_CONTEXT_RESULTS = 10  
        GRADING_MODE = False
        ENABLE_QUERY_ROUTER = True
        ROUTER_HISTORY_WINDOW = 4
        ENABLE_HYDE = False  # generate hypothetical answer to improve retrieval
        ENABLE_MULTI_QUERY = False
        ENABLE_CONTEXTUAL_COMPRESSION = False
        ENABLE_QUERY_DECOMPOSITION = False
        DECOMPOSE_MAX_SUBQUESTIONS = 3
        DECOMPOSE_MIN_WORDS = 10
        ENABLE_QUERY_SCORING = False
        QUERY_SCORE_TOP_K = 3
        QUERY_SCORE_WEIGHTS = {"semantic": 0.7, "lexical": 0.3}
        QUERY_SCORE_THRESHOLDS = {"high": 0.75, "medium": 0.55}
        COMPRESSION_MIN_RATIO = 0.8
        COMPRESSION_MAX_DOC_LENGTH = 1200
        DEBUG_LLM_CONTEXT = False
        DEBUG_CONTEXT_FILEPATH = "llm_context_debug.txt"

        # Neighbor Expansion: expand retrieved chunks to include adjacent chunks
        # solves: section headers at page end, content split by tables/figures, mid-sentence breaks
        ENABLE_NEIGHBOR_EXPANSION = True
        NEIGHBOR_MAX_PER_DIRECTION = 1  # max neighbors to add per direction (forward/backward)
        NEIGHBOR_MIN_OVERLAP_RATIO = 0.15  # minimum content overlap to trigger expansion (15%)
        NEIGHBOR_ENABLE_FORWARD = True  # expand to next chunks
        NEIGHBOR_ENABLE_BACKWARD = True  # expand to previous chunks
        NEIGHBOR_DEBUG = False  # print expansion debug info
        MAX_FINAL_CONTEXT_CHUNKS = 16  # Reduced from 16 to prevent context overflow (16K limit)
        PREFILTER_BEFORE_RERANK = 20   # Limit docs sent to reranker (RRF gives 30+, rerank only top 20)

    class Eval:
        """Evaluation settings"""
        GOLD_SET_PATH = "eval/gold_set.json"
        K = 4  # recall@k
        OUTPUT_DIR = "eval/results"
        ENABLE_LLM_JUDGE = True
        JUDGE_MODEL = "chatbot_llm"  # Uses same LLM as chatbot (shared singleton)
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
         EMBEDDING_BATCH_SIZE = 4  # batch size for embedding generation (conservative for 6GB GPU)
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
        ENABLE_STRATIFIED_TABLE_RETRIEVAL = True  # separate lane for tables with neutral weight
        # candidate counts for the table-only retriever
        N_TABLE_SEMANTIC_RESULTS = 3
        N_TABLE_BM25_RESULTS = 3
        # how strongly table-only results influence fused ranking
        TABLE_LIST_WEIGHT = 1.2 
        # reranker guardrail 
        MIN_TABLES_IN_CONTEXT = 2

    class Raptor:
        """
        RAPTOR Lite: Hierarchical summarization for improved multi-hop retrieval.
        Creates cluster summaries that provide high-level context.
        """
        # number of clusters (None = auto-calculate using sqrt(n/2) heuristic)
        NUM_CLUSTERS = None
        # cluster size constraints
        MIN_CLUSTER_SIZE = 3   # skip clusters smaller than this
        MAX_CLUSTER_SIZE = 50  # split large clusters
        # summary generation
        SUMMARY_MODEL = None  # None = use chatbot LLM