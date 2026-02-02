import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config
from llama_wrapper import ChatLlamaCppWrapper


@dataclass
class DocumentMetadata:
    """Document-level metadata extracted from full document"""
    title: str
    doc_type: str  # research_paper, financial_report, textbook, manual, article, other
    total_pages: int
    total_chunks: int
    has_tables: bool
    has_figures: bool
    has_equations: bool
    language: str  # en, es, fr, etc.
    estimated_tokens: int

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ChunkMetadata:
    """Chunk-level metadata for a single text chunk"""
    # position metadata
    chunk_index: int
    page: int
    section: Optional[str]

    # content metadata
    content_type: str  # text, table, figure, equation
    char_count: int
    estimated_tokens: int
    has_math: bool
    has_code: bool
    has_urls: bool

    # quality metadata
    is_complete: bool  # no cut-off sentences
    information_density: str  # low, medium, high

    def to_dict(self) -> Dict:
        return asdict(self)

DOCUMENT_TYPE_PROMPT = ChatPromptTemplate.from_template(
    """Classify this document into ONE category based on its content:

Categories:
- research_paper: academic/scientific papers with abstract, methodology, results
- financial_report: earnings reports, financial statements, investor documents
- textbook: educational material with chapters, exercises, explanations
- technical_manual: user guides, API documentation, technical specifications
- article: blog posts, news articles, opinion pieces
- legal_document: contracts, terms of service, legal filings
- other: anything that doesn't fit above categories

Document preview (first 2000 chars):
{preview}

Output ONLY the category name, nothing else."""
)

SECTION_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """Extract the section/chapter heading that appears before this text chunk.
Rules:
- look for headings like "Chapter 3", "Introduction", "3.2 Methods", etc.
- return ONLY the heading text, no extra explanation
- if no clear heading found, return "None"

Document context (500 chars before chunk):
{context_before}

Chunk:
{chunk}

Section heading:"""
)

class MetadataExtractor:
    """
    Extracts rich metadata from documents and chunks

    Uses heuristics + optional LLM calls for classification
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None

        if use_llm:
            # create LLM for metadata extraction using wrapper
            model_path = str(Config.Model.GGUF_PATH.resolve())
            self.llm = ChatLlamaCppWrapper(
                model_path=model_path,
                temperature=0,  # deterministic
                n_ctx=1024,   # small - just need to extract doc type/language
                max_tokens=50, # very short outputs
                n_gpu_layers=0,  # CPU only to save VRAM
                n_batch=Config.Model.N_BATCH,
                n_threads=Config.Model.N_THREADS,
                streaming=False,
                verbose=False,
            )
            self.doc_type_chain = DOCUMENT_TYPE_PROMPT | self.llm | StrOutputParser()
            self.section_chain = SECTION_EXTRACTION_PROMPT | self.llm | StrOutputParser()

    def cleanup(self):
        """Free LLM resources"""
        if self.llm is not None:
            del self.llm
            self.llm = None
            import gc
            gc.collect()

    def __del__(self):
        """Cleanup on garbage collection"""
        self.cleanup()

    def extract_document_metadata(
        self,
        full_text: str,
        filename: str,
        chunks: List[Document]
    ) -> DocumentMetadata:
        """
        Extract document-level metadata
        """
        # detect document type
        doc_type = self._classify_document_type(full_text)
        # extract title (first meaningful line or filename)
        title = self._extract_title(full_text, filename)
        # count pages (from chunks metadata)
        pages = set()
        for chunk in chunks:
            if 'page' in chunk.metadata:
                pages.add(chunk.metadata['page'])
        total_pages = len(pages) if pages else 1
        # detect content types
        has_tables = any(c.metadata.get('content_type') == 'table' for c in chunks)
        has_figures = any(c.metadata.get('content_type') == 'figure' for c in chunks)
        has_equations = self._has_equations(full_text)

        # language detection (simple heuristic)
        language = self._detect_language(full_text)

        # estimate tokens (rough: 1 token ~= 4 chars)
        estimated_tokens = len(full_text) // 4
        return DocumentMetadata(
            title=title,
            doc_type=doc_type,
            total_pages=total_pages,
            total_chunks=len(chunks),
            has_tables=has_tables,
            has_figures=has_figures,
            has_equations=has_equations,
            language=language,
            estimated_tokens=estimated_tokens,
        )

    def enrich_chunk_metadata(
        self,
        chunk: Document,
        chunk_index: int,
        full_document: str = None
    ) -> Document:
        """
        Enrich a chunk with additional metadata
        """
        metadata = chunk.metadata.copy()

        # basic metadata
        metadata['chunk_index'] = chunk_index

        # content analysis
        text = chunk.page_content
        metadata['char_count'] = len(text)
        metadata['estimated_tokens'] = len(text) // 4
        metadata['has_math'] = self._has_math(text)
        metadata['has_code'] = self._has_code(text)
        metadata['has_urls'] = self._has_urls(text)

        # quality assessment
        metadata['is_complete'] = self._is_complete_chunk(text)
        metadata['information_density'] = self._assess_information_density(text)
        # section extraction (if LLM enabled and full doc available)
        if self.use_llm and full_document and metadata.get('content_type') == 'text':
            section = self._extract_section(chunk, full_document)
            if section and section.lower() != 'none':
                metadata['section'] = section

        chunk.metadata = metadata
        return chunk

    def _classify_document_type(self, text: str) -> str:
        """Classify document type using heuristics or LLM"""
        preview = text[:2000]

        # fast heuristic classification
        lower_preview = preview.lower()

        # financial indicators
        if any(term in lower_preview for term in ['earnings', 'revenue', 'fiscal year', 'balance sheet', 'income statement']):
            return 'financial_report'

        # academic paper indicators
        if any(term in lower_preview for term in ['abstract', 'introduction', 'methodology', 'references', 'citation']):
            if 'chapter' not in lower_preview:  # distinguish from textbooks
                return 'research_paper'

        # textbook indicators
        if any(term in lower_preview for term in ['chapter', 'exercise', 'problem set', 'learning objectives']):
            return 'textbook'

        # technical manual indicators
        if any(term in lower_preview for term in ['installation', 'configuration', 'api reference', 'user guide']):
            return 'technical_manual'

        # legal document indicators
        if any(term in lower_preview for term in ['whereas', 'hereby', 'terms and conditions', 'agreement']):
            return 'legal_document'

        # use LLM as fallback if enabled
        if self.use_llm:
            try:
                doc_type = self.doc_type_chain.invoke({'preview': preview}).strip().lower()
                # clean thinking tags
                if '</think>' in doc_type:
                    doc_type = doc_type.split('</think>')[-1].strip().lower()

                valid_types = ['research_paper', 'financial_report', 'textbook',
                              'technical_manual', 'article', 'legal_document', 'other']
                if doc_type in valid_types:
                    return doc_type
            except Exception as e:
                print(f"    LLM classification failed: {e}")

        return 'other'

    def _extract_title(self, text: str, filename: str) -> str:
        """Extract document title from first lines or use filename"""
        lines = text.split('\n')

        # look for first substantial line (>10 chars, <200 chars)
        for line in lines[:20]:
            clean_line = line.strip()
            if 10 < len(clean_line) < 200:
                # skip if it looks like metadata
                if not re.match(r'^(page|author|date|copyright)', clean_line.lower()):
                    return clean_line

        # fallback to filename without extension
        return filename.rsplit('.', 1)[0]

    def _detect_language(self, text: str) -> str:
        """
        language detection (currently defaults to english)

        multi-language support will be added in future
        """
        return 'en'

    def _has_equations(self, text: str) -> bool:
        """Detect if document has mathematical equations"""
        # look for LaTeX math or equation indicators
        patterns = [
            r'\$.*?\$',  # inline math
            r'\\\[.*?\\\]',  # display math
            r'\\begin\{equation\}',
            r'\\frac\{',
            r'\\sum_',
            r'\\int_',
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _has_math(self, text: str) -> bool:
        """Detect if chunk contains mathematical notation"""
        # mathematical symbols and patterns
        math_indicators = [
            r'\$.*?\$',  # LaTeX
            r'[α-ωΑ-Ω]',  # Greek letters
            r'[∑∫∂∇≈≠≤≥±×÷]',  # math symbols
            r'\b\d+\s*[+\-*/=]\s*\d+',  # equations
            r'\\frac|\\sum|\\int',  # LaTeX commands
        ]
        return any(re.search(pattern, text) for pattern in math_indicators)

    def _has_code(self, text: str) -> bool:
        """Detect if chunk contains code snippets"""
        # code indicators
        code_patterns = [
            r'```',  # markdown code blocks
            r'def\s+\w+\s*\(',  # python function
            r'function\s+\w+\s*\(',  # javascript function
            r'public\s+class\s+\w+',  # java class
            r'import\s+\w+',  # import statement
            r'=>',  # arrow function
            r'\{\s*\n',  # opening brace on new line
        ]
        return any(re.search(pattern, text) for pattern in code_patterns)

    def _has_urls(self, text: str) -> bool:
        """Detect if chunk contains URLs"""
        url_pattern = r'https?://[^\s]+'
        return bool(re.search(url_pattern, text))

    def _is_complete_chunk(self, text: str) -> bool:
        """Assess if chunk is complete (no cut-off sentences)"""
        # check if ends with sentence-ending punctuation
        text_stripped = text.rstrip()
        if not text_stripped:
            return False

        # ends with period, question mark, or exclamation
        if text_stripped[-1] in '.?!':
            return True

        # ends with closing bracket/quote after punctuation
        if len(text_stripped) >= 2 and text_stripped[-1] in '")]}' and text_stripped[-2] in '.?!':
            return True

        return False

    def _assess_information_density(self, text: str) -> str:
        """Assess information density: low, medium, high"""
        # heuristics based on text characteristics
        words = text.split()
        if not words:
            return 'low'

        # calculate metrics
        avg_word_length = sum(len(w) for w in words) / len(words)
        sentence_count = text.count('.') + text.count('?') + text.count('!')
        words_per_sentence = len(words) / max(sentence_count, 1)

        # high density: long words, complex sentences, numbers/symbols
        has_numbers = bool(re.search(r'\d', text))
        has_technical_terms = avg_word_length > 6
        complex_sentences = words_per_sentence > 20

        if has_technical_terms and (has_numbers or complex_sentences):
            return 'high'
        elif avg_word_length > 5 or words_per_sentence > 15:
            return 'medium'
        else:
            return 'low'

    def _extract_section(self, chunk: Document, full_document: str) -> Optional[str]:
        """Extract section heading for chunk using LLM"""
        if not self.llm:
            return None

        try:
            # find chunk position in document
            chunk_text = chunk.page_content[:200]  # use start of chunk
            chunk_pos = full_document.find(chunk_text)

            if chunk_pos == -1:
                return None

            # get context before chunk (look for section heading)
            context_start = max(0, chunk_pos - 500)
            context_before = full_document[context_start:chunk_pos]

            # call LLM to extract section
            section = self.section_chain.invoke({
                'context_before': context_before,
                'chunk': chunk_text
            }).strip()

            # clean thinking tags
            if '</think>' in section:
                section = section.split('</think>')[-1].strip()

            return section if section.lower() != 'none' else None

        except Exception as e:
            print(f"Section extraction failed: {e}")
            return None


# global instance
_global_extractor: Optional[MetadataExtractor] = None

def get_metadata_extractor(use_llm: bool = True) -> MetadataExtractor:
    """Get or create global metadata extractor instance"""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = MetadataExtractor(use_llm=use_llm)
    return _global_extractor
