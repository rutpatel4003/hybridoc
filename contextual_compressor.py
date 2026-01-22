from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from config import Config


EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """Extract ONLY the sentences from the document that are relevant to answering the question.

RULES:
- Include complete sentences, not fragments
- Preserve exact wording (no paraphrasing)
- Include key facts, numbers, and technical terms
- If the entire document is relevant, return it as-is
- If nothing is relevant, return "NOT_RELEVANT"
- Output only the extracted text, no explanations

Question: {question}

Document:
{document}

Extracted relevant portions:"""
)


class ContextualCompressor:
    """
    Compresses retrieved documents by extracting only relevant portions.

    This runs AFTER reranking but BEFORE generation, providing:
    1. Reduced noise in context
    2. Faster generation (fewer tokens)
    3. More focused answers
    """

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0,
        max_doc_length: int = 2000,
        min_compression_ratio: float = 0.3,
        enable: bool = True
    ):
        self.model_name = model_name or Config.Model.NAME
        self.temperature = temperature
        self.max_doc_length = max_doc_length
        self.min_compression_ratio = min_compression_ratio
        self.enable = enable

        if self.enable:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                num_ctx=4096,
                num_predict=512,
                keep_alive=-1,
            )
            self.chain = EXTRACTION_PROMPT | self.llm | StrOutputParser()

    def _should_compress(self, doc: Document) -> bool:
        """Determine if document should be compressed"""
        # don't compress tables/figures (they're already concise)
        content_type = doc.metadata.get('content_type', 'text')
        if content_type in ('table', 'figure'):
            return False

        # only compress long documents
        return len(doc.page_content) > self.max_doc_length

    def _compress_single_doc(
        self,
        doc: Document,
        question: str
    ) -> Document:
        """Compress a single document"""
        if not self._should_compress(doc):
            return doc

        try:
            # Extract relevant portions
            compressed = self.chain.invoke({
                'question': question,
                'document': doc.page_content[:4000]  # limit input to LLM
            })

            # clean thinking tags if present
            if '</think>' in compressed:
                compressed = compressed.split('</think>')[-1].strip()

            # check if extraction returned nothing relevant
            if compressed.strip() == "NOT_RELEVANT" or len(compressed) < 50:
                return doc  # Return original if extraction failed

            # check compression ratio
            ratio = len(compressed) / len(doc.page_content)
            if ratio > self.min_compression_ratio:
                # not much compression achieved, return original
                return doc

            # return compressed version
            return Document(
                page_content=compressed,
                metadata={
                    **doc.metadata,
                    'compressed': True,
                    'original_length': len(doc.page_content),
                }
            )

        except Exception as e:
            print(f"  Compression failed: {e}")
            return doc  # fallback to original

    def compress(
        self,
        docs: List[Document],
        question: str
    ) -> List[Document]:
        """
        Compress multiple documents by extracting relevant portions.
        """
        if not self.enable or not docs:
            return docs

        compressed_docs = []
        original_chars = sum(len(d.page_content) for d in docs)

        for doc in docs:
            compressed_doc = self._compress_single_doc(doc, question)
            compressed_docs.append(compressed_doc)

        # calculate compression stats
        compressed_chars = sum(len(d.page_content) for d in compressed_docs)
        compression_ratio = compressed_chars / original_chars if original_chars > 0 else 1.0
        chars_saved = original_chars - compressed_chars

        print(f"  Compression: {len(docs)} docs, "
              f"{original_chars} → {compressed_chars} chars "
              f"({compression_ratio:.1%} retained, saved {chars_saved} chars)")

        return compressed_docs


# global compressor instance
_global_compressor: ContextualCompressor = None


def get_compressor() -> ContextualCompressor:
    """Get or create global compressor instance"""
    global _global_compressor
    if _global_compressor is None:
        from config import Config
        enable_compression = getattr(Config.Chatbot, 'ENABLE_CONTEXTUAL_COMPRESSION', False)
        _global_compressor = ContextualCompressor(enable=enable_compression)
    return _global_compressor
