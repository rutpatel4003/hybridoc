# HybriDoc

![Status](https://img.shields.io/badge/Status-Production--Ready-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Ollama-orange)
![VectorDB](https://img.shields.io/badge/VectorDB-Qdrant-red)

Production-grade RAG system with vision-based table extraction, hybrid retrieval, and automated evaluation. Runs entirely on-premise with GPU acceleration.

## Key Features

**Vision-Based Document Processing**
- Qwen3-VL-4B extracts tables as images (handles graphics-rendered financial statements)
- Flash Attention 2 for 20-30% faster inference with lower VRAM
- Structured table parsing with OCR artifact normalization (β 1 = 0.9 → β1 = 0.9)
- Persistent disk caching (MD5-based) eliminates redundant OCR passes

**Production Retrieval Pipeline**
- Hybrid search: BM25 + semantic embeddings with Reciprocal Rank Fusion
- Parent-child chunking (500 char retrieval → 3072 char context)
- Cross-encoder reranking (top-3 from 12 candidates)
- Semantic query cache: 90%+ similarity threshold, 50-80% latency reduction
- Source routing for multi-file scenarios (keyword-based, zero VRAM overhead)
- Query-aware table row scoring (lexical for <50 rows, semantic for larger tables)

**Query Processing**
- Multi-turn conversation tracking (standalone/follow-up/chitchat routing)
- Query decomposition for multi-hop questions
- HyDE (Hypothetical Document Embeddings) for abstract queries
- LLM-as-Judge evaluation with automated faithfulness scoring

## Architecture

```
PDF Upload → PyMuPDF Parser → Caption Detection → Qwen3-VL Vision OCR
                  ↓                                       ↓
            Text Blocks                            Tables/Figures
                  ↓                                       ↓
            Chunking + Metadata              Structured Parsing
                  ↓                                       ↓
              Qdrant Vector DB + BM25 Index
                           ↓
        Query → Cache Check → Hybrid Retrieval → Reranking
                     ↓              ↓
                Cache Hit      RRF Fusion
                     ↓              ↓
                Retrieved Documents
                         ↓
               Ollama LLM → Stream Response
```

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **UI** | Streamlit | Real-time streaming, table rendering |
| **LLM** | Ollama (Qwen3:4b-instruct) | Local inference, no API costs |
| **Vision** | Qwen3-VL-4B-Instruct (4-bit) | Flash Attention 2 enabled |
| **Embeddings** | Alibaba-NLP/gte-multilingual-base | 768-dim semantic vectors |
| **Vector DB** | Qdrant | Persistent, filtered search |
| **Keyword Search** | BM25 (rank-bm25) | Built from Qdrant corpus |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L12-v2 | Context refinement |
| **Evaluation** | LLM-as-Judge (Qwen3:4b-thinking) | Automated quality checks |

## Performance

**Latency** (70-page financial report, RTX 3060)
| Operation | Time |
|-----------|------|
| PDF Ingestion (15-20 tables) | 3-5 min |
| Query (cold) | 800-1200ms |
| Query (cached) | 200-400ms |
| Reindexing (disk cache hit) | 10-15s |

**Resource Usage**
| Component | VRAM |
|-----------|------|
| Qwen3-VL-4B (4-bit) | 4-6 GB |
| Embeddings (GTE-base) | 768 MB |
| Ollama (Qwen3:4b) | 2-3 GB |

## Benchmarks

_TBD: Metrics will be added after finalizing evaluation dataset size and document selection._

## Installation

**Prerequisites**
- Python 3.10+
- CUDA-capable GPU (recommended, CPU fallback supported)
- Ollama ([ollama.com](https://ollama.com))

```bash
# Clone repository
git clone https://github.com/rutpatel4003/hybridoc.git
cd hybridoc

# Setup environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt

# Pull Ollama models
ollama pull qwen3:4b-instruct
ollama pull qwen3:4b-thinking  # Optional: for evaluation
```

## Usage

```bash
streamlit run app.py
```

1. Upload documents (PDF/Markdown/Text) via sidebar
2. Wait for indexing (progress bar displayed)
3. Ask questions: *"What is the dropout rate in Table 2?"* or *"Compare revenue in Q1 vs Q2"*

**Interface Features**
- Interactive tables with DataFrame view and CSV export
- Token-by-token streaming responses
- Source citations with page numbers and content types

## Configuration

Core settings in `config.py`:

```python
# Retrieval
CHUNK_SIZE = 2000
N_SEMANTIC_RESULTS = 6
N_BM25_RESULTS = 6
N_CONTEXT_RESULTS = 3  # After reranking

# Features
ENABLE_QUERY_DECOMPOSITION = True
ENABLE_PARENT_CHILD = False
CONTEXTUALIZE_CHUNKS = False
ENABLE_TABLE_SEMANTIC_ENRICHMENT = True

# Performance
ENABLE_QUERY_CACHE = True
CACHE_SIMILARITY_THRESHOLD = 0.90
DEVICE = 'cuda'  # Auto-detected, or override to 'cpu'
```

## Evaluation

```bash
python -m eval.run_eval --pdf docs/report.pdf --k 4
```

Outputs retrieval metrics (Hit Rate, Recall@k, MRR), latency distribution, and markdown report to `eval/results/`.

**Gold Set Format** (`eval/gold_set.json`):
```json
{
  "questions": [{
    "id": "q1",
    "question": "What is the total revenue in Q1 2025?",
    "expected_sources": ["2025-q1-report.pdf"],
    "expected_pages": [6],
    "keywords": ["revenue", "123"],
    "expected_content_type": "table"
  }]
}
```

## Project Structure

```
hybridoc/
├── app.py                      # Streamlit UI
├── chatbot.py                  # LangGraph workflow, query routing
├── data_ingestor.py           # Chunking, indexing, retrieval pipeline
├── pdf_loader.py              # PDF parsing, Qwen3-VL OCR
├── docling_loader.py          # Alternative loader with Docling
├── vector_store_qdrant.py     # Qdrant wrapper
├── table_intelligence.py      # Table parsing, OCR normalization
├── table_enricher.py          # LLM table descriptions
├── metadata_extractor.py      # Document/chunk metadata
├── query_cache.py             # Semantic caching
├── source_router.py           # Multi-file source detection
├── neighbor_expansion.py      # Context expansion retriever
├── config.py                  # Configuration
├── eval/
│   ├── evaluator.py          # Retrieval + faithfulness metrics
│   ├── llm_judge.py          # LLM-as-Judge
│   └── run_eval.py           # CLI runner
└── requirements.txt
```

## Limitations

- Multi-document temporal reasoning requires explicit query decomposition
- Mathematical equations OCR'd as text, not symbolic representation
- PDFs >200 pages may require batching for GPU memory constraints

## License

MIT License

## Acknowledgments

Built with Qwen3-VL, Alibaba GTE embeddings, Qdrant, and Ollama.
