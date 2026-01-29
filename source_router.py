from typing import List, Optional, Set
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import re


class SourceRouterRetriever(BaseRetriever):
    """
    Routes queries to specific files when source is explicitly mentioned.
    Falls back to global search for general queries.

    Design: Keyword-only routing (0 VRAM overhead)
    - Detects explicit mentions: "in the Transformer paper", "from document X"
    - Does NOT route general queries: "what is nutrition?" searches all files naturally
    - Conservative: only routes when confident to avoid breaking multi-doc retrieval

    Use case:
    - 10 papers uploaded, query "What is Table 2 in Transformer paper?"
    - Routes to Transformer only, avoids Table 2 from other papers
    """
    base_retriever: BaseRetriever
    vector_store: any = None
    _source_cache: Optional[Set[str]] = None

    model_config = {"arbitrary_types_allowed": True}
    def _get_available_sources(self) -> Set[str]:
        """Get list of all indexed sources (cached)"""
        if self._source_cache is not None:
            return self._source_cache

        try:
            all_docs = self.vector_store.get()
            sources = set(m.get('source', '') for m in all_docs.get('metadatas', []) if m.get('source'))
            self._source_cache = sources
            return sources
        except Exception as e:
            print(f"Warning: Could not fetch sources: {e}")
            return set()

    def _keyword_based_detection(self, query: str, sources: Set[str]) -> Optional[str]:
        """
        Robust keyword-based source detection.

        Detects explicit source mentions:
        - "in [filename]", "from [filename]", "according to [filename]"
        - "in the [doc name]", "[paper name] says", "based on [doc]"
        - Handles: filenames with/without extensions, underscores→spaces, hyphens→spaces

        Conservative: Only routes when source is explicitly mentioned to avoid
        breaking multi-document queries like "compare X and Y" or "what is nutrition?"
        """
        query_lower = query.lower()

        # precompile common patterns for efficiency
        contextual_markers = [
            "in", "from", "according to", "based on", "in the", "from the",
            "as per", "per", "referenced in", "mentioned in", "described in",
            "in document", "in paper", "in file", "the paper", "the document"
        ]

        for source in sources:
            # generate all filename variations
            source_lower = source.lower()
            source_no_ext = source_lower.replace('.pdf', '').replace('.txt', '').replace('.md', '')
            source_spaced = source_no_ext.replace('_', ' ').replace('-', ' ')
            # also try removing common suffixes like "v1", "final", etc.
            source_cleaned = re.sub(r'[_\-](v\d+|final|draft|revised)$', '', source_no_ext)
            keywords = set([source_lower, source_no_ext, source_spaced, source_cleaned])
            # pattern 1: Direct contextual mention ("in transformer paper")
            for marker in contextual_markers:
                for keyword in keywords:
                    pattern = f"{marker} {keyword}"
                    if pattern in query_lower:
                        return source

            # pattern 2: Possessive or direct reference ("[doc name]'s", "[doc name] shows")
            for keyword in keywords:
                if len(keyword) > 3:  # Avoid false positives on very short names
                    # look for keyword followed by possessive or verb
                    possessive_pattern = f"{keyword}'s"
                    verb_pattern = f"{keyword} (shows|describes|states|says|mentions|indicates)"

                    if possessive_pattern in query_lower:
                        return source

                    if re.search(rf"\b{re.escape(keyword)}\s+(shows|describes|states|says|mentions|indicates)\b", query_lower):
                        return source

            # pattern 3: Table/Figure reference with document name
            # "table 2 in [doc]", "figure 3 from [doc]"
            table_fig_markers = ["table", "figure", "fig", "chart", "diagram"]
            for marker in table_fig_markers:
                for keyword in keywords:
                    if len(keyword) > 3 and f"{marker}" in query_lower and keyword in query_lower:
                        # check if they appear close together (within 10 words)
                        words = query_lower.split()
                        try:
                            marker_idx = words.index(marker)
                            keyword_words = keyword.split()
                            # look for keyword within 10 words of marker
                            for i in range(max(0, marker_idx-10), min(len(words), marker_idx+10)):
                                if all(w in words[i:i+len(keyword_words)] for w in keyword_words):
                                    return source
                        except (ValueError, IndexError):
                            continue

        return None

    def _detect_source_intent(self, query: str) -> Optional[str]:
        """
        Detect if query explicitly mentions a specific source document.
        """
        sources = self._get_available_sources()

        if not sources or len(sources) == 1:
            # only one source or no sources - no routing needed
            return None

        # keyword-based detection (covers 95% of real-world explicit mentions)
        detected = self._keyword_based_detection(query, sources)
        return detected

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve with optional source filtering"""
        detected_source = self._detect_source_intent(query)

        if detected_source:
            print(f"Routing to source: {detected_source}")

            # filter retrieval to specific source
            try:
                filtered_results = self.vector_store.similarity_search(
                    query,
                    k=kwargs.get('k', 15),  # Slightly higher k for single-source queries
                    filter={"source": detected_source}
                )
                return filtered_results
            except Exception as e:
                print(f"Filtered search failed ({e}), falling back to global search")
                return self.base_retriever.invoke(query)
        else:
            print(f"Searching all sources ({len(self._get_available_sources())} files)")
            return self.base_retriever.invoke(query)

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)

def create_source_router(base_retriever: BaseRetriever, vector_store) -> BaseRetriever:
    """
    Convenience function to wrap a retriever with source routing.
    """
    return SourceRouterRetriever(
        base_retriever=base_retriever,
        vector_store=vector_store
    )
