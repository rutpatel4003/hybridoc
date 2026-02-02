"""
Full RAG Evaluation: Retrieval + Generation + Faithfulness

Runs complete evaluation including:
- Retrieval accuracy (Hit Rate, Recall@k, MRR)
- Answer quality (Faithfulness, Relevance, Completeness)
- Hallucination detection
- Latency metrics

Usage:
    python -m eval.run_full_eval --pdf ../file1.pdf --pdf ../file2.pdf --gold-set eval/gold_set_5docs_90q.json
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Any
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from eval.evaluator import Evaluator, EvalConfig
from chatbot import Chatbot
from data_ingestor import ingest_files
from config import Config
from pdf_loader import File


def load_pdf_file(pdf_path: str) -> File:
    """
    Load PDF file for evaluation (CLI version, no Streamlit dependencies).
    Replicates the logic from pdf_loader.load_uploaded_file but for file paths.
    """
    from pathlib import Path
    import hashlib
    
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Check cache first (same logic as pdf_loader)
    file_bytes = path_obj.read_bytes()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    cache_dir = Config.Path.DATA_DIR / "ocr_cache"
    cache_path = cache_dir / f"{file_hash}.txt"
    cache_blocks_path = cache_dir / f"{file_hash}_blocks.json"
    
    if cache_path.exists():
        print(f"  Cache hit: {path_obj.name}")
        content = cache_path.read_text(encoding='utf-8')
        
        # Try to load structured blocks
        content_blocks = None
        if cache_blocks_path.exists():
            import json
            try:
                from table_intelligence import StructuredTable
                from pdf_loader import ContentBlock
                
                blocks_data = json.loads(cache_blocks_path.read_text(encoding='utf-8'))
                content_blocks = []
                for block_dict in blocks_data:
                    table_data = None
                    if block_dict.get('table_data'):
                        td = block_dict['table_data']
                        table_data = StructuredTable(
                            headers=td['headers'],
                            rows=td['rows'],
                            raw_markdown=td['raw_markdown'],
                            num_rows=td['num_rows'],
                            num_cols=td['num_cols']
                        )
                    content_blocks.append(ContentBlock(
                        content=block_dict['content'],
                        content_type=block_dict['content_type'],
                        page_num=block_dict['page_num'],
                        bbox=tuple(block_dict['bbox']) if block_dict.get('bbox') else None,
                        table_data=table_data,
                        caption_label=block_dict.get('caption_label'),
                        section_header=block_dict.get('section_header')
                    ))
            except Exception as e:
                print(f"    Warning: Could not load cached blocks: {e}")
        
        return File(name=path_obj.name, content=content, content_blocks=content_blocks)
    
    # Process fresh (replicates pdf_loader.load_uploaded_file logic)
    print(f"  Processing: {path_obj.name}")
    
    if path_obj.suffix.lower() == '.pdf':
        if getattr(Config.Preprocessing, 'USE_DOCLING', True):
            from docling_loader import extract_pdf_with_docling_from_bytes, format_docling_blocks_as_text
            content_blocks = extract_pdf_with_docling_from_bytes(
                file_bytes,
                filename=path_obj.name,
                use_ocr=getattr(Config.Preprocessing, 'USE_DOCLING_OCR', True)
            )
            content = format_docling_blocks_as_text(content_blocks)
        else:
            from pdf_loader import extract_pdf_content_with_structure, format_content_blocks_as_text
            content_blocks = extract_pdf_content_with_structure(file_bytes, use_got_ocr=True)
            content = format_content_blocks_as_text(content_blocks)
    else:
        # Markdown or text
        content = file_bytes.decode('utf-8')
        content_blocks = None
    
    # Save to cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding='utf-8')
        
        if content_blocks:
            import json
            blocks_data = []
            for block in content_blocks:
                block_dict = {
                    'content': block.content,
                    'content_type': block.content_type,
                    'page_num': block.page_num,
                    'bbox': list(block.bbox) if block.bbox else None,
                    'table_data': block.table_data.to_dict() if block.table_data else None,
                    'caption_label': block.caption_label,
                    'section_header': block.section_header
                }
                blocks_data.append(block_dict)
            cache_blocks_path.write_text(json.dumps(blocks_data, indent=2), encoding='utf-8')
    except Exception as e:
        print(f"    Warning: Could not cache: {e}")
    
    return File(name=path_obj.name, content=content, content_blocks=content_blocks)


class PipelineRetriever(BaseRetriever):
    """
    Wrapper that uses the EXACT Chatbot retrieval pipeline including:
    - Query condensing (_condense_question)
    - Query decomposition (_decompose_question inside _retrieve)
    - Neighbor expansion
    - Parent expansion  
    - Contextual compression
    - Query scoring
    - HyDE support (_hyde_retrieve)
    
    This ensures evaluation matches production behavior exactly.
    """
    
    chatbot: Any = None  # Declare as Pydantic field
    
    model_config = {"arbitrary_types_allowed": True}  # Allow arbitrary types like Chatbot
    
    def __init__(self, chatbot: Chatbot):
        # Call super().__init__ with the field value instead of direct assignment
        super().__init__(chatbot=chatbot)
    
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Replicates the LangGraph workflow logic from chatbot.py exactly.
        """
        # Initialize state exactly as the workflow does
        state = {
            'question': query,
            'chat_history': [],  # Empty for single-turn eval
            'retry_count': 0,
            'context': [],
            'sub_questions': [],
            'confidence': None
        }
        
        # Step 1: Condense question (if query router enabled)
        if Config.Chatbot.ENABLE_QUERY_ROUTER:
            try:
                # Merge like LangGraph does (don't overwrite entire state)
                result = self.chatbot._condense_question(state)
                state.update(result)
                if state['question'] != query:
                    print(f"  Condensed: '{state['question'][:80]}...'")
            except Exception as e:
                print(f"  Condense warning: {e}")
        
        # Step 2: Retrieve (HyDE or Normal) - EXACTLY as chatbot.py does
        if Config.Chatbot.ENABLE_HYDE:
            print("  Using HyDE retrieval")
            result = self.chatbot._hyde_retrieve(state)
        else:
            result = self.chatbot._retrieve(state)
        
        # Extract documents from result
        docs = result.get('context', [])
        
        # Log pipeline results (matching chatbot.py behavior)
        confidence = result.get('confidence')
        if confidence:
            print(f"  Confidence: {confidence.get('label', 'unknown')} ({confidence.get('score', 0):.2f})")
        
        return docs
    
    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Async fallback"""
        return self._get_relevant_documents(query, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Full RAG evaluation with answer generation and faithfulness')
    parser.add_argument('--pdf', action='append', required=True, help='Path to PDF file(s)')
    parser.add_argument('--gold-set', default='eval/gold_set.json', help='Path to gold set JSON')
    parser.add_argument('--k', type=int, default=4, help='Recall@k')
    parser.add_argument('--hyde', action='store_true', help='Enable HyDE')
    parser.add_argument('--no-hyde', action='store_true', help='Disable HyDE')
    parser.add_argument('--verbose', action='store_true', help='Show judge reasoning')
    args = parser.parse_args()

    # Determine HyDE (command line overrides config)
    use_hyde = Config.Chatbot.ENABLE_HYDE
    if args.no_hyde:
        use_hyde = False
    if args.hyde:
        use_hyde = True

    print("="*80)
    print("🧪 FULL RAG EVALUATION (Retrieval + Generation + Faithfulness)")
    print("="*80)
    print(f"Gold Set: {args.gold_set}")
    print(f"Recall@k: k={args.k}")
    print(f"HyDE: {'ENABLED' if use_hyde else 'DISABLED'}")
    print(f"Query Router: {'ENABLED' if Config.Chatbot.ENABLE_QUERY_ROUTER else 'DISABLED'}")
    print(f"Query Decomposition: {'ENABLED' if Config.Chatbot.ENABLE_QUERY_DECOMPOSITION else 'DISABLED'}")
    print(f"Neighbor Expansion: {'ENABLED' if Config.Chatbot.ENABLE_NEIGHBOR_EXPANSION else 'DISABLED'}")
    print(f"Parent-Child: {'ENABLED' if Config.Preprocessing.ENABLE_PARENT_CHILD else 'DISABLED'}")
    print(f"Contextual Retrieval: {'ENABLED' if Config.Preprocessing.ENABLE_CONTEXTUAL_RETRIEVAL else 'DISABLED'}")
    print(f"Contextual Compression: {'ENABLED' if Config.Chatbot.ENABLE_CONTEXTUAL_COMPRESSION else 'DISABLED'}")
    print(f"Max Final Context: {Config.Chatbot.MAX_FINAL_CONTEXT_CHUNKS} chunks")
    print(f"Judge: {Config.Model.GGUF_PATH.name} (shared with chatbot)")
    print()

    # Load PDFs
    print("="*80)
    print("📄 LOADING PDF FILES...")
    print("="*80)
    files = []
    for pdf_path in args.pdf:
        try:
            file = load_pdf_file(pdf_path)
            files.append(file)
            print(f"  ✓ Loaded {file.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {pdf_path}: {e}")
            sys.exit(1)
    print()

    # Temporarily override HyDE config for this run
    original_hyde = Config.Chatbot.ENABLE_HYDE
    Config.Chatbot.ENABLE_HYDE = use_hyde

    # Create chatbot (this calls ingest_files internally)
    print("="*80)
    print("💬 CREATING CHATBOT...")
    print("="*80)
    chatbot = Chatbot(files)
    print("✓ Chatbot ready\n")

    # Restore original config
    Config.Chatbot.ENABLE_HYDE = original_hyde

    # Create the PipelineRetriever that wraps the full chatbot logic
    # CRITICAL: This uses _retrieve/_hyde_retrieve methods with all pipeline features
    # instead of just chatbot.retriever (which is only the base RRF/BM25 layer)
    print("="*80)
    print("🔧 EVALUATION SETUP")
    print("="*80)
    print("Using FULL PIPELINE retriever (includes decomposition, expansion, compression)")
    pipeline_retriever = PipelineRetriever(chatbot)
    
    # Run evaluation with the pipeline retriever
    print("="*80)
    print("🧪 RUNNING FULL EVALUATION...")
    print("="*80)
    print()

    config = EvalConfig(k=args.k, gold_set_path=Path(args.gold_set))
    evaluator = Evaluator(pipeline_retriever, config)

    detailed_results, aggregate_metrics = evaluator.evaluate_with_faithfulness(
        chatbot,
        verbose=args.verbose
    )

    # Save results
    results_dir = Path('eval/results')
    results_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    import json
    results_file = results_dir / f"eval_full_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'k': args.k, 
                'hyde': use_hyde,
                'query_router': Config.Chatbot.ENABLE_QUERY_ROUTER,
                'query_decomposition': Config.Chatbot.ENABLE_QUERY_DECOMPOSITION,
                'neighbor_expansion': Config.Chatbot.ENABLE_NEIGHBOR_EXPANSION,
                'parent_child': Config.Preprocessing.ENABLE_PARENT_CHILD,
                'contextual_compression': Config.Chatbot.ENABLE_CONTEXTUAL_COMPRESSION,
                'gold_set': args.gold_set
            },
            'aggregate': aggregate_metrics,
            'results': detailed_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Results saved to: {results_file}\n")

    # Print final summary
    print("="*80)
    print("📈 FINAL METRICS")
    print("="*80)
    print(f"\n📊 RETRIEVAL")
    print(f"  Hit Rate:     {aggregate_metrics.get('retrieval_hit_rate', 0):.1%}")
    print(f"  Recall@{args.k}:     {aggregate_metrics.get('mean_recall_at_k', 0):.1%}")
    print(f"  MRR:          {aggregate_metrics.get('mean_reciprocal_rank', 0):.3f}")
    print(f"\n✨ ANSWER QUALITY")
    print(f"  Faithfulness: {aggregate_metrics.get('mean_faithfulness', 0):.1%}")
    print(f"  Relevance:    {aggregate_metrics.get('mean_relevance', 0):.1%}")
    print(f"  Overall:      {aggregate_metrics.get('mean_overall_quality', 0):.1%}")
    print(f"\n  SAFETY")
    print(f"  Hallucination: {aggregate_metrics.get('hallucination_rate', 0):.1%}")
    print(f"  Below Thresh:  {aggregate_metrics.get('answers_below_faith_threshold', 0):.1%}")
    print(f"\n⚡ PERFORMANCE")
    print(f"  Avg Latency:   {aggregate_metrics.get('mean_latency_ms', 0):.0f}ms")
    print("="*80)


if __name__ == "__main__":
    main()