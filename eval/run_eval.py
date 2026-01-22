import argparse
import sys
from pathlib import Path
from typing import List

# add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_loader import File
from data_ingestor import ingest_files
from eval.evaluator import run_evaluation
from config import Config

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


HYDE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant. Given a question, write a short passage (2-3 sentences)
that would directly answer the question. Write as if this passage exists in a technical document.

Question: {question}

Passage:"""
)


class HyDERetriever(BaseRetriever):
    """
    Wrapper that adds HyDE (Hypothetical Document Embeddings) to any retriever.
    """
    base_retriever: BaseRetriever
    llm: ChatOllama = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.llm is None:
            self.llm = ChatOllama(
                model=Config.Model.NAME,
                temperature=0,
                num_ctx=2048,
                num_predict=256,
                keep_alive=0,  # Unload model after each call to save VRAM
            )

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Generate hypothetical doc, then retrieve"""
        try:
            # Generate hypothetical answer
            hyde_chain = HYDE_PROMPT | self.llm | StrOutputParser()
            hypothetical = hyde_chain.invoke({'question': query})

            # Clean thinking tags if model outputs them
            if '</think>' in hypothetical:
                hypothetical = hypothetical.split('</think>')[-1].strip()

            print(f"  HyDE: {hypothetical[:80]}...")

            # Retrieve using hypothetical document
            hyde_results = self.base_retriever.invoke(hypothetical)

            # Also get normal results and merge
            normal_results = self.base_retriever.invoke(query)

            # Deduplicate, prioritizing HyDE results
            seen = set()
            merged = []
            for doc in hyde_results + normal_results:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen:
                    merged.append(doc)
                    seen.add(content_hash)

            return merged

        except Exception as e:
            print(f"  HyDE failed ({e}), using normal retrieval")
            return self.base_retriever.invoke(query)

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--gold-set", default="eval/gold_set.json", help="Path to gold set JSON")
    parser.add_argument("--k", type=int, default=4, help="k for Recall@k")
    parser.add_argument("--pdf", action="append", help="PDF files to index (can specify multiple)")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE even if enabled in config")
    parser.add_argument("--hyde", action="store_true", help="Force enable HyDE")
    parser.add_argument("--multi-query", action="store_true", help="Enable multi-query expansion")
    parser.add_argument("--no-multi-query", action="store_true", help="Disable multi-query expansion")
    args = parser.parse_args()

    if not args.pdf:
        print("No PDFs specified. Use --pdf path/to/file.pdf")
        print("Example: python -m eval.run_eval --pdf docs/paper.pdf --k 4")
        sys.exit(1)

    # load files
    print("Loading files...")
    files = []
    for pdf_path in args.pdf:
        path = Path(pdf_path)
        if not path.exists():
            print(f"File not found: {pdf_path}")
            sys.exit(1)

        with open(path, 'rb') as f:
            content = f.read()

        from pdf_loader import extract_pdf_content_with_structure, format_content_blocks_as_text
        blocks = extract_pdf_content_with_structure(content)
        text = format_content_blocks_as_text(blocks)
        files.append(File(name=path.name, content=text, content_blocks=blocks))

    print(f"Loaded {len(files)} file(s)")

    # build retriever
    print("Building retriever...")
    retriever = ingest_files(files)

    # Determine if HyDE should be used
    use_hyde = Config.Chatbot.ENABLE_HYDE
    if args.no_hyde:
        use_hyde = False
    if args.hyde:
        use_hyde = True

    if use_hyde:
        print("🔮 HyDE enabled - wrapping retriever")
        retriever = HyDERetriever(base_retriever=retriever)
    else:
        print("📊 HyDE disabled")

    # Determine if multi-query should be used
    use_multi_query = Config.Chatbot.ENABLE_MULTI_QUERY
    if args.no_multi_query:
        use_multi_query = False
    if args.multi_query:
        use_multi_query = True

    if use_multi_query and not use_hyde:  # Don't stack with HyDE
        print("🔄 Multi-Query enabled - generating query variations")
        from data_ingestor import MultiQueryRetriever
        retriever = MultiQueryRetriever(base_retriever=retriever)
    elif not use_multi_query:
        print("📊 Multi-Query disabled")

    # Print config summary
    print(f"\n📋 Config:")
    print(f"   HyDE: {'ON' if use_hyde else 'OFF'}")
    print(f"   Multi-Query: {'ON' if use_multi_query and not use_hyde else 'OFF'}")
    print(f"   Parent-Child: {'ON' if Config.Preprocessing.ENABLE_PARENT_CHILD else 'OFF'}")
    print(f"   k={args.k}")
    print()

    # run evaluation
    run_evaluation(retriever, gold_set_path=args.gold_set, k=args.k)


if __name__ == "__main__":
    main()
