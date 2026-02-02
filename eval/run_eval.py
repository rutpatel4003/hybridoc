import argparse
import sys
from pathlib import Path
from typing import List
import hashlib
import re
from datetime import datetime

# add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_loader import File, extract_pdf_content_with_structure, format_content_blocks_as_text
from data_ingestor import ingest_files, get_chatbot_llm, expand_to_parents, create_embeddings
from eval.evaluator import run_evaluation
from config import Config

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def log_gpu(label=""):
    """Log GPU memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU {label}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    except:
        pass


def load_pdf_file(pdf_path: str) -> File:
    """
    Load a PDF file using EXACT SAME pipeline as main chatbot (pdf_loader.py:580-650)
    Uses Docling + Surya OCR (primary) with fallback to PyMuPDF + GOT-OCR
    """
    from pathlib import Path

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Loading {pdf_path.name}...")
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()

    content_blocks = None

    # Try Docling first (same as pdf_loader.py:586)
    if getattr(Config.Preprocessing, 'USE_DOCLING', True):
        try:
            from docling_loader import extract_pdf_with_docling_from_bytes, format_docling_blocks_as_text
            print(f"  🔧 Mode: Docling + Surya OCR (primary)")

            use_ocr = getattr(Config.Preprocessing, 'USE_DOCLING_OCR', True)
            content_blocks = extract_pdf_with_docling_from_bytes(
                file_bytes,
                filename=pdf_path.name,
                use_ocr=use_ocr
            )
            content = format_docling_blocks_as_text(content_blocks)

            # Check if Docling found tables (same as pdf_loader.py:600-630)
            tables_found = sum(1 for b in content_blocks if b.content_type == 'table')
            if tables_found > 0:
                print(f"  ✓ Docling extracted {tables_found} tables successfully")
            else:
                print(f"  ⚠ Docling returned 0 tables. Running native table pass...")
                # Fallback to native extraction for tables
                native_blocks = extract_pdf_content_with_structure(file_bytes, use_got_ocr=False)
                native_tables = [b for b in native_blocks if b.content_type == "table"]

                if native_tables:
                    print(f"  ✓ Native extraction found {len(native_tables)} tables. Merging...")
                    # Merge Docling text blocks + native tables
                    text_blocks = [b for b in content_blocks if b.content_type != "table"]
                    content_blocks = text_blocks + native_tables
                    content = format_content_blocks_as_text(content_blocks)

        except Exception as e:
            print(f"  ⚠ Docling failed: {e}")
            print(f"  Falling back to native extraction...")
            content_blocks = None

    # Fallback to native extraction (same as pdf_loader.py:640)
    if content_blocks is None:
        print(f"  🔧 Mode: PyMuPDF + GOT-OCR (fallback)")
        content_blocks = extract_pdf_content_with_structure(file_bytes, use_got_ocr=True)
        content = format_content_blocks_as_text(content_blocks)

    return File(name=pdf_path.name, content=content, content_blocks=content_blocks)


def _stable_doc_uid(doc: Document) -> str:
    """Generate stable unique ID for a document (same as chatbot.py)"""
    md = doc.metadata or {}
    source = md.get('source', '')
    page = md.get('page', 0)
    content_hash = hashlib.md5(doc.page_content[:500].encode()).hexdigest()[:8]
    return f"{source}:{page}:{content_hash}"


def _is_explicit_question(text: str) -> bool:
    """
    Return True if the input is already a clear standalone question (chatbot.py:567-574)
    """
    text = text.strip()
    if not text.endswith('?'):
        return False
    return bool(re.search(r"\b(what|why|how|when|where|which|who|compare|difference|define|explain)\b", text.lower()))


def condense_query(question: str, chat_history=None) -> str:
    """
    Simplified query router for eval (replicates chatbot.py:_condense_question)
    For eval: All questions are single-turn with no history, so just check if explicit
    """
    if not Config.Chatbot.ENABLE_QUERY_ROUTER:
        return question

    # Already a clear question - no condensing needed
    if _is_explicit_question(question):
        return question

    # For eval, no chat history, so treat as standalone
    return question


# EXACT SAME HYDE_PROMPT AS CHATBOT (chatbot.py:178-192)
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical document author. Given a question, write a short paragraph (3-5 sentences) that would appear in a document answering this question.

RULES:
- Write as if you are the document, not answering directly
- Include technical terms and specific details that would appear in the actual document
- Do NOT say "The document explains..." — just write the content itself
- Keep it factual and dense with keywords

Example:
Question: "What is the time complexity of quicksort?"
Output: "Quicksort has an average-case time complexity of O(n log n) and a worst-case complexity of O(n²). The algorithm uses a divide-and-conquer approach, selecting a pivot element and partitioning the array. Space complexity is O(log n) due to recursive stack frames."
"""),
    ("human", "{question}"),
])


# EXACT SAME DECOMPOSE_PROMPT AS CHATBOT (chatbot.py:162-176)
DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given a question, break it into 1-3 simpler sub-questions that cover different aspects.

Return ONLY a JSON array of strings. No explanations.

Examples:
Input: "Compare the performance of model A and model B"
Output: ["What is the performance of model A?", "What is the performance of model B?"]

Input: "What is the architecture and training procedure?"
Output: ["What is the architecture?", "What is the training procedure?"]

Input: "What is the capital of France?"
Output: ["What is the capital of France?"]"""),
    ("human", "{question}"),
])


class NormalRetrievalWrapper(BaseRetriever):
    """
    Replicates Chatbot._retrieve (chatbot.py:450-522) - Normal retrieval path
    - Query decomposition
    - Multi-query retrieval with stable deduplication
    - Neighbor expansion
    - Parent expansion
    - Contextual compression (if enabled)
    - Query scoring (if enabled)
    """
    base_retriever: BaseRetriever
    llm: object = None  # ChatLlamaCpp instance
    query_scorer: object = None  # QueryScorer instance (initialized once)

    model_config = {"arbitrary_types_allowed": True}

    def _is_complex_query(self, question: str) -> bool:
        """
        EXACT SAME as chatbot.py:524-531
        Heuristic for detecting complex/multi-hop queries
        """
        if len(question.split()) < Config.Chatbot.DECOMPOSE_MIN_WORDS:
            return False
        # EXACT keywords from chatbot.py:530
        keywords = ["compare", "difference", "between", "versus", "vs", "both", "and", "contrast", "relative"]
        return any(k in question.lower() for k in keywords)

    def _decompose_question(self, question: str) -> List[str]:
        """Same decomposition logic as chatbot.py:533-565"""
        if not Config.Chatbot.ENABLE_QUERY_DECOMPOSITION:
            return [question]

        if not self._is_complex_query(question):
            return [question]

        try:
            import json
            chain = DECOMPOSE_PROMPT | self.llm | StrOutputParser()
            raw = chain.invoke({"question": question})

            # Clean thinking tags
            if '</think>' in raw:
                raw = raw.split('</think>')[-1].strip()

            # Parse JSON
            sub_questions = json.loads(raw)

            # Sanitize
            cleaned = []
            for q in sub_questions:
                q = q.strip()
                if not q.endswith('?'):
                    q += '?'
                cleaned.append(q)

            # Remove duplicates & limit
            unique = list(dict.fromkeys(cleaned))
            max_sub = getattr(Config.Chatbot, 'DECOMPOSE_MAX_SUBQUESTIONS', 3)
            result = unique[:max_sub]

            if len(result) > 1:
                print(f"  Decomposed into {len(result)} sub-questions")
            return result

        except Exception as e:
            print(f"  Decomposition failed: {e}")
            return [question]

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """EXACT SAME LOGIC as Chatbot._retrieve (chatbot.py:450-522)"""
        print(f"RETRIEVING: {query}")

        # 1. Decompose query (chatbot.py:454)
        sub_questions = self._decompose_question(query)

        # 2. Retrieve with each sub-question + stable dedupe (chatbot.py:456-476)
        all_docs: List[Document] = []
        seen: dict[str, int] = {}

        def _is_table(d: Document) -> bool:
            md = d.metadata or {}
            return str(md.get("content_type", "")).lower() == "table"

        for q in sub_questions:
            docs = self.base_retriever.invoke(q)
            for doc in docs:
                uid = _stable_doc_uid(doc)

                if uid not in seen:
                    seen[uid] = len(all_docs)
                    all_docs.append(doc)
                    continue

                # Prefer keeping TABLE over non-table (chatbot.py:471-476)
                existing_idx = seen[uid]
                existing_doc = all_docs[existing_idx]
                if _is_table(doc) and not _is_table(existing_doc):
                    all_docs[existing_idx] = doc

        context = all_docs

        # 3. Neighbor expansion (chatbot.py:479-496)
        if getattr(Config.Chatbot, 'ENABLE_NEIGHBOR_EXPANSION', False):
            try:
                from neighbor_expansion import expand_neighbors
                original_count = len(context)
                context = expand_neighbors(
                    retrieved_docs=context,
                    retriever=self.base_retriever,
                    max_neighbors=getattr(Config.Chatbot, 'NEIGHBOR_MAX_PER_DIRECTION', 1),
                    min_overlap_ratio=getattr(Config.Chatbot, 'NEIGHBOR_MIN_OVERLAP_RATIO', 0.15),
                    enable_forward=getattr(Config.Chatbot, 'NEIGHBOR_ENABLE_FORWARD', True),
                    enable_backward=getattr(Config.Chatbot, 'NEIGHBOR_ENABLE_BACKWARD', True),
                    debug=getattr(Config.Chatbot, 'NEIGHBOR_DEBUG', False)
                )
                if len(context) > original_count:
                    print(f"  Neighbor expansion: {original_count} → {len(context)} chunks (+{len(context) - original_count})")
            except Exception as e:
                print(f"  Neighbor expansion failed: {e}, continuing without it")

        # 4. Parent expansion (chatbot.py:498-501)
        if Config.Preprocessing.ENABLE_PARENT_CHILD:
            try:
                original_count = len(context)
                context = expand_to_parents(context)
                if len(context) > original_count:
                    print(f"  Expanded {original_count} children → {len(context)} parents")
            except Exception as e:
                print(f"  Parent expansion failed: {e}")

        # 5. Contextual compression (chatbot.py:503-506)
        if getattr(Config.Chatbot, 'ENABLE_CONTEXTUAL_COMPRESSION', False):
            try:
                from contextual_compressor import get_compressor
                compressor = get_compressor(llm=self.llm)
                context = compressor.compress(context, query)
                print(f"  Contextual compression applied")
            except Exception as e:
                print(f"  Contextual compression failed: {e}")

        # 6. Final context limit (chatbot.py:524-529)
        max_final = getattr(Config.Chatbot, 'MAX_FINAL_CONTEXT_CHUNKS', 16)
        if len(context) > max_final:
            original_len = len(context)
            context = context[:max_final]
            print(f"  Final limit: {original_len} → {max_final} chunks (capped)")

        # 7. Query scoring (chatbot.py:531-541) - for metrics only
        if getattr(Config.Chatbot, 'ENABLE_QUERY_SCORING', False):
            try:
                if self.query_scorer is None:
                    from query_scoring import QueryScorer
                    self.query_scorer = QueryScorer(embedder=create_embeddings())
                score = self.query_scorer.score(query, context)
                print(f"  Query confidence: {score.label} ({score.score:.2f})")
            except Exception as e:
                print(f"  Query scoring failed: {e}")

        return context

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


class HyDERetrievalWrapper(BaseRetriever):
    """
    Replicates Chatbot._hyde_retrieve (chatbot.py:576-619) - HyDE retrieval path
    - Generate hypothetical document
    - Retrieve with hypothetical doc + original query
    - Merge and deduplicate
    - Parent expansion
    - Contextual compression (if enabled)

    NOTE: Does NOT do query decomposition (HyDE generates full answers, not questions)
    """
    base_retriever: BaseRetriever
    llm: object = None  # ChatLlamaCpp instance

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """EXACT SAME LOGIC as Chatbot._hyde_retrieve (chatbot.py:576-619)"""
        print(f'HyDE RETRIEVING: {query}')

        try:
            # 1. Generate hypothetical document (chatbot.py:587-592)
            hyde_chain = HYDE_PROMPT | self.llm | StrOutputParser()
            hypothetical_doc = hyde_chain.invoke({'question': query})

            # Clean thinking tags
            if '</think>' in hypothetical_doc:
                hypothetical_doc = hypothetical_doc.split('</think>')[-1].strip()

            print(f"HyDE generated: {hypothetical_doc[:100]}...")

            # 2. Retrieve with hypothetical document (chatbot.py:595)
            context = self.base_retriever.invoke(hypothetical_doc)

            # 3. Normal retrieval with original query (chatbot.py:597)
            normal_context = self.base_retriever.invoke(query)

            # 4. Merge and deduplicate (chatbot.py:599-604)
            seen_content = set()
            merged = []
            for doc in context + normal_context:
                if doc.page_content not in seen_content:
                    merged.append(doc)
                    seen_content.add(doc.page_content)

            # 5. Parent expansion (chatbot.py:605-606)
            if Config.Preprocessing.ENABLE_PARENT_CHILD:
                try:
                    original_count = len(merged)
                    merged = expand_to_parents(merged)
                    if len(merged) > original_count:
                        print(f"  Expanded {original_count} children → {len(merged)} parents")
                except Exception as e:
                    print(f"  Parent expansion failed: {e}")

            # 6. Limit to configured max (chatbot.py:608)
            merged = merged[:Config.Chatbot.N_CONTEXT_RESULTS * 2]

            print(f"HyDE retrieved {len(context)} + normal {len(normal_context)} = {len(merged)} unique docs")

            # 7. Contextual compression (chatbot.py:611-614)
            if getattr(Config.Chatbot, 'ENABLE_CONTEXTUAL_COMPRESSION', False):
                try:
                    from contextual_compressor import get_compressor
                    compressor = get_compressor(llm=self.llm)
                    merged = compressor.compress(merged, query)
                    print(f"  Contextual compression applied")
                except Exception as e:
                    print(f"  Contextual compression failed: {e}")

            return merged

        except Exception as e:
            print(f"HyDE failed ({e}), falling back to normal retrieval")
            context = self.base_retriever.invoke(query)
            return context

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


class CondensedRetriever(BaseRetriever):
    """
    Wrapper that applies query condensing before retrieval (chatbot.py:823-880)
    For eval: Single-turn queries with no chat history
    """
    base_retriever: BaseRetriever
    chat_history: List = []

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Apply query condensing (replicates chatbot._condense_question)
        condensed = condense_query(query, self.chat_history)
        if condensed != query:
            print(f"  ROUTER: Condensed '{query[:50]}...' → '{condensed[:50]}...'")
        return self.base_retriever.invoke(condensed, **kwargs)

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Run Private-RAG evaluation')
    parser.add_argument('--pdf', action='append', required=True, help='Path to PDF file(s) to index')
    parser.add_argument('--gold-set', default='eval/gold_set.json', help='Path to gold set JSON')
    parser.add_argument('--k', type=int, default=4, help='Number of documents to retrieve')
    parser.add_argument('--hyde', action='store_true', help='Enable HyDE')
    parser.add_argument('--no-hyde', action='store_true', help='Disable HyDE')
    parser.add_argument('--output', default=None, help='Output log file (default: eval/logs/eval_TIMESTAMP.log)')
    args = parser.parse_args()

    # Setup output logging
    if args.output is None:
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output = str(log_dir / f'eval_{timestamp}.log')

    # Tee output to both console and file
    import io
    class TeeOutput(io.StringIO):
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(args.output, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, log_file)

    print(f"📝 Logging output to: {args.output}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load PDF files
    print("\n" + "="*80)
    print("📄 LOADING PDF FILES...")
    print("="*80)
    files = []
    for pdf_path in args.pdf:
        try:
            file_obj = load_pdf_file(pdf_path)
            files.append(file_obj)
            print(f"  ✓ Loaded {file_obj.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {pdf_path}: {e}")
            sys.exit(1)
    print(f"✓ Loaded {len(files)} PDF files")

    # Configuration verification (CRITICAL)
    print("\n" + "="*80)
    print("⚙️  CONFIGURATION VERIFICATION")
    print("="*80)
    use_hyde = Config.Chatbot.ENABLE_HYDE
    if args.no_hyde:
        use_hyde = False
    if args.hyde:
        use_hyde = True

    print(f"Retrieval Mode:")
    print(f"  HyDE: Config={Config.Chatbot.ENABLE_HYDE}, Args Override={use_hyde}")
    print(f"\nQuery Processing:")
    print(f"  Query Router: {Config.Chatbot.ENABLE_QUERY_ROUTER}")
    print(f"  Query Decomposition: {Config.Chatbot.ENABLE_QUERY_DECOMPOSITION}")
    print(f"  Query Scoring: {Config.Chatbot.ENABLE_QUERY_SCORING}")
    print(f"\nRetrieval Enhancement:")
    print(f"  Neighbor Expansion: {Config.Chatbot.ENABLE_NEIGHBOR_EXPANSION}")
    print(f"  Parent-Child Chunking: {Config.Preprocessing.ENABLE_PARENT_CHILD}")
    print(f"  Contextual Compression: {getattr(Config.Chatbot, 'ENABLE_CONTEXTUAL_COMPRESSION', False)}")
    print(f"  Max Final Context: {Config.Chatbot.MAX_FINAL_CONTEXT_CHUNKS} chunks")
    print(f"  Cross-Page Bridging: {Config.Preprocessing.ENABLE_CROSS_PAGE_BRIDGING}")
    print(f"\nTable Handling:")
    print(f"  Stratified Table Retrieval: {Config.Tables.ENABLE_STRATIFIED_TABLE_RETRIEVAL}")
    print(f"  Table Semantic Enrichment: {Config.Preprocessing.ENABLE_TABLE_SEMANTIC_ENRICHMENT}")
    print(f"\nIndexing:")
    print(f"  Contextual Retrieval: {Config.Preprocessing.ENABLE_CONTEXTUAL_RETRIEVAL}")
    print(f"  Metadata Extraction: {Config.Preprocessing.ENABLE_METADATA_EXTRACTION}")

    if Config.Chatbot.ENABLE_QUERY_ROUTER:
        print(f"\n⚠️  Query Router is ENABLED - eval queries will be condensed (single-turn, minimal impact)")

    # Build base retriever (same as Chatbot.__init__)
    print("\n" + "="*80)
    print("🤖 BUILDING BASE RETRIEVER...")
    print("="*80)
    log_gpu("Before ingest_files")
    base_retriever = ingest_files(files)
    log_gpu("After ingest_files")

    # Get LLM (same as Chatbot.__init__)
    llm = get_chatbot_llm()
    print(f"✓ LLM loaded: {llm}")

    # Wrap with EITHER NormalRetrievalWrapper OR HyDERetrievalWrapper (mutually exclusive)
    if use_hyde:
        print("🔮 HyDE enabled - using HyDERetrievalWrapper")
        retriever = HyDERetrievalWrapper(base_retriever=base_retriever, llm=llm)
    else:
        print("📊 HyDE disabled - using NormalRetrievalWrapper")
        retriever = NormalRetrievalWrapper(base_retriever=base_retriever, llm=llm)

    # Wrap with CondensedRetriever (outermost layer - applies query condensing first)
    if Config.Chatbot.ENABLE_QUERY_ROUTER:
        print("🔀 Query Router enabled - wrapping with CondensedRetriever")
        retriever = CondensedRetriever(base_retriever=retriever, chat_history=[])
    else:
        print("🔀 Query Router disabled - skipping condensation")

    # Run evaluation
    print("\n" + "="*80)
    print("🧪 RUNNING EVALUATION...")
    print("="*80)
    run_evaluation(retriever, gold_set_path=args.gold_set, k=args.k)

    print(f"\n📝 Full log saved to: {args.output}")

    # Close log file
    sys.stdout = original_stdout
    log_file.close()


if __name__ == '__main__':
    main()
