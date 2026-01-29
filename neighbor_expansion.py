from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
import re

class NeighborExpander:
    """
    Expands retrieved chunks to include adjacent chunks when semantic continuity is detected.
    Integrated with existing Qdrant vector store and metadata structure.
    """
    def __init__(
        self,
        vector_store,  # QdrantVectorStore instance
        max_neighbors: int = 1,
        min_overlap_ratio: float = 0.15,
        enable_forward: bool = True,
        enable_backward: bool = True,
        debug: bool = False
    ):
        self.vector_store = vector_store
        self.max_neighbors = max_neighbors
        self.min_overlap_ratio = min_overlap_ratio
        self.enable_forward = enable_forward
        self.enable_backward = enable_backward
        self.debug = debug
        # cache for source documents (avoid repeated queries)
        self._source_cache: Dict[str, List[Document]] = {}
        
    def expand(self, retrieved_docs: List[Document]) -> List[Document]:
        """
        Main expansion logic.
        """
        if not retrieved_docs:
            return []
        # group by source document
        doc_groups = self._group_by_source(retrieved_docs)

        expanded = []
        expansion_stats = {"added": 0, "skipped": 0, "cache_hits": 0}
        MAX_TOTAL_ADDITIONS = 8

        for source, docs in doc_groups.items():
            current_additions = len(expanded) - len(retrieved_docs)
            if current_additions >= MAX_TOTAL_ADDITIONS:
                break
            source_expanded = self._expand_source_group(source, docs, expansion_stats)
            expanded.extend(source_expanded)
        
        # deduplicate while preserving order
        deduped = self._deduplicate(expanded)
        if self.debug:
            added = len(deduped) - len(retrieved_docs)
            print(f"Neighbor Expansion: {len(retrieved_docs)} → {len(deduped)} chunks (+{added})")
            print(f"Stats: {expansion_stats['added']} added, {expansion_stats['skipped']} skipped, "
                  f"{expansion_stats['cache_hits']} cache hits")
        # clear cache after expansion
        self._source_cache.clear()
        return deduped
    
    def _group_by_source(self, docs: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by source file."""
        groups: Dict[str, List[Document]] = {}
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in groups:
                groups[source] = []
            groups[source].append(doc)
        return groups
    
    def _expand_source_group(
        self, 
        source: str, 
        docs: List[Document],
        stats: Dict[str, int]
    ) -> List[Document]:
        """Expand chunks from a single source document."""
        expanded = list(docs)  # start with originals
        seen_chunks = {self._get_chunk_key(d) for d in docs}
        
        for doc in docs:
            # check expansion triggers
            needs_forward, needs_backward = self._check_expansion_triggers(doc)
            
            # skip if no expansion needed
            if not needs_forward and not needs_backward:
                continue
            
            # backward expansion
            if needs_backward and self.enable_backward:
                neighbors = self._get_neighbors(doc, direction='backward', stats=stats)
                for neighbor in neighbors[:self.max_neighbors]:
                    key = self._get_chunk_key(neighbor)
                    if key not in seen_chunks:
                        if self._validate_continuity(doc, neighbor, 'backward'):
                            expanded.append(neighbor)
                            seen_chunks.add(key)
                            stats['added'] += 1
                        else:
                            stats['skipped'] += 1
            
            # forward expansion
            if needs_forward and self.enable_forward:
                neighbors = self._get_neighbors(doc, direction='forward', stats=stats)
                for neighbor in neighbors[:self.max_neighbors]:
                    key = self._get_chunk_key(neighbor)
                    if key not in seen_chunks:
                        if self._validate_continuity(doc, neighbor, 'forward'):
                            expanded.append(neighbor)
                            seen_chunks.add(key)
                            stats['added'] += 1
                        else:
                            stats['skipped'] += 1
        return expanded
    
    def _check_expansion_triggers(self, doc: Document) -> Tuple[bool, bool]:
        """
        Detect if a chunk should be expanded based on boundary signals.
        """
        content = doc.page_content.strip()
        metadata = doc.metadata or {}
        
        needs_forward = False
        needs_backward = False
        # check for section header at end
        if self._ends_with_header(content):
            needs_forward = True
        # check for incomplete sentences
        if self._starts_mid_sentence(content):
            needs_backward = True
        
        if self._ends_mid_sentence(content):
            needs_forward = True
        # check for cross-page bridge marker
        if '[CROSS_PAGE_BRIDGE]' in content:
            needs_forward = True
            needs_backward = True
        # short text chunks near page boundaries likely need context
        content_type = metadata.get('content_type', 'text')
        if len(content) < 200 and content_type == 'text':
            needs_forward = True
            needs_backward = True
        
        return needs_forward, needs_backward
    
    def _ends_with_header(self, content: str) -> bool:
        """Check if content ends with a section header."""
        lines = content.strip().split('\n')
        if not lines:
            return False
        
        last_line = lines[-1].strip()
        
        # pattern: "5.4 Regularization" or "## Introduction"
        header_patterns = [
            r'^\d+(\.\d+)*\s+[A-Z][^.!?]*$',  # 5.4 Regularization
            r'^#{1,6}\s+[A-Z]',  # Markdown headers
            r'^[A-Z][^.!?]{5,50}$',  # Short capitalized lines without punctuation
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, last_line):
                return True
        return False
    
    def _starts_mid_sentence(self, content: str) -> bool:
        """Check if content starts mid-sentence."""
        content = content.lstrip()
        if not content:
            return False
        
        # lowercase start suggests continuation
        if content[0].islower():
            return True
        
        # starts with conjunction/continuation word
        continuation_words = ['and', 'but', 'or', 'however', 'therefore', 'thus', 'hence']
        first_word = content.split()[0].lower().strip('.,;:')
        if first_word in continuation_words:
            return True
        
        return False
    
    def _ends_mid_sentence(self, content: str) -> bool:
        """Check if content ends mid-sentence."""
        content = content.rstrip()
        if not content:
            return False
        
        # missing terminal punctuation
        if content[-1] not in '.!?':
            return True
        # ends with comma or semicolon
        if content[-1] in ',;:':
            return True
        return False
    
    def _get_neighbors(
        self, 
        doc: Document, 
        direction: str = 'forward',
        stats: Optional[Dict[str, int]] = None
    ) -> List[Document]:
        """
        Fetch adjacent chunks from vector store.
        Uses cached source documents to avoid repeated queries.
        """
        metadata = doc.metadata or {}
        source = metadata.get('source', '')
        
        if not source:
            return []
        
        # get all chunks from this source (with caching)
        all_source_docs = self._get_source_documents(source, stats)
        
        if not all_source_docs:
            return []
        
        # find adjacent chunks
        return self._find_adjacent_chunks(doc, all_source_docs, direction)
    
    def _get_source_documents(self, source: str, stats: Optional[Dict[str, int]] = None) -> List[Document]:
        """
        Get all chunks from a source document, with caching.
        """
        # check cache first
        if source in self._source_cache:
            if stats:
                stats['cache_hits'] += 1
            return self._source_cache[source]
        
        # query vector store
        try:
            # use vector_store.get() with source filter
            result = self.vector_store.get(filter={"source": source})
            # reconstruct Document objects from result
            documents = []
            texts = result.get('documents', [])
            metadatas = result.get('metadatas', [])
            
            for text, metadata in zip(texts, metadatas):
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
            
            # cache for future use
            self._source_cache[source] = documents
            
            return documents
            
        except Exception as e:
            if self.debug:
                print(f"Failed to fetch source documents for {source}: {e}")
            return []
    
    def _find_adjacent_chunks(
        self,
        anchor: Document,
        all_docs: List[Document],
        direction: str
    ) -> List[Document]:
        """
        Find adjacent chunks based on chunk_index and page.
        """
        anchor_meta = anchor.metadata or {}
        anchor_page = anchor_meta.get('page', anchor_meta.get('page_number', 0))
        anchor_index = anchor_meta.get('chunk_index', anchor_meta.get('chunk_id', -1))
        
        # convert to int
        try:
            anchor_page = int(anchor_page)
            anchor_index = int(anchor_index)
        except (ValueError, TypeError):
            return []
        # filter to same or adjacent pages
        candidates = []
        for doc in all_docs:
            doc_meta = doc.metadata or {}
            doc_page = doc_meta.get('page', doc_meta.get('page_number', 0))
            doc_index = doc_meta.get('chunk_index', doc_meta.get('chunk_id', -1))
            
            try:
                doc_page = int(doc_page)
                doc_index = int(doc_index)
            except (ValueError, TypeError):
                continue
            
            # skip if not adjacent page
            if abs(doc_page - anchor_page) > 1:
                continue
            
            # filter by direction
            if direction == 'forward' and doc_index > anchor_index:
                candidates.append((doc_index, doc))
            elif direction == 'backward' and doc_index < anchor_index:
                candidates.append((doc_index, doc))
        
        # sort by index
        if direction == 'forward':
            candidates.sort(key=lambda x: x[0])  # ascending
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)  # descending
        
        return [doc for _, doc in candidates]
    
    def _validate_continuity(
        self,
        anchor: Document,
        neighbor: Document,
        direction: str
    ) -> bool:
        """
        Validate structural continuity (no lexical overlap check for general-purpose RAG).
        """
        anchor_meta = anchor.metadata or {}
        neighbor_meta = neighbor.metadata or {}
        # must be from same source
        if anchor_meta.get('source') != neighbor_meta.get('source'):
            return False

        # pages should be adjacent or same
        anchor_page = anchor_meta.get('page', anchor_meta.get('page_number', 0))
        neighbor_page = neighbor_meta.get('page', neighbor_meta.get('page_number', 0))

        try:
            if abs(int(anchor_page) - int(neighbor_page)) > 1:
                return False
        except (ValueError, TypeError):
            pass

        # check for hard boundaries (chapter breaks)
        if self._has_hard_boundary(neighbor):
            return False

        return True

    def _has_hard_boundary(self, doc: Document) -> bool:
        """Check if document starts with a hard boundary (chapter, major section)."""
        content = doc.page_content.strip()
        
        # pattern: "Chapter 5" or "Part II"
        boundary_patterns = [
            r'^Chapter\s+\d+',
            r'^Part\s+[IVX]+',
            r'^Section\s+[IVX]+',
        ]
        
        for pattern in boundary_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _get_chunk_key(self, doc: Document) -> str:
        """Generate unique key for chunk deduplication."""
        metadata = doc.metadata or {}
        source = metadata.get('source', '')
        page = metadata.get('page', metadata.get('page_number', ''))
        chunk_index = metadata.get('chunk_index', metadata.get('chunk_id', ''))
        
        # use chunk_hash if available (from vector store)
        chunk_hash = metadata.get('chunk_hash', '')
        if not chunk_hash:
            chunk_hash = hash(doc.page_content)
        
        return f"{source}|{page}|{chunk_index}|{chunk_hash}"
    
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """Deduplicate while preserving order."""
        seen = set()
        deduped = []
        
        for doc in docs:
            key = self._get_chunk_key(doc)
            if key not in seen:
                seen.add(key)
                deduped.append(doc)
        
        return deduped


def expand_neighbors(
    retrieved_docs: List[Document],
    retriever,  # retriever with .vector_store attribute
    max_neighbors: int = 1,
    min_overlap_ratio: float = 0.15,
    enable_forward: bool = True,
    enable_backward: bool = True,
    debug: bool = False
) -> List[Document]:
    """
    Convenience function for neighbor expansion.
    """
    # extract vector_store from retriever
    vector_store = None
    
    # try different ways to get vector_store
    if hasattr(retriever, 'vector_store'):
        vector_store = retriever.vector_store
    elif hasattr(retriever, 'base_retriever'):
        # handle wrapped retrievers (e.g., ParentChildRetriever)
        base = retriever.base_retriever
        if hasattr(base, 'vector_store'):
            vector_store = base.vector_store
        # try deeper nesting
        while hasattr(base, 'base_retriever') and not vector_store:
            base = base.base_retriever
            if hasattr(base, 'vector_store'):
                vector_store = base.vector_store
                break
    elif hasattr(retriever, 'retrievers'):
        # handle ensemble retrievers
        for r in retriever.retrievers:
            if hasattr(r, 'vector_store'):
                vector_store = r.vector_store
                break
    # fallback: try to find in child_retriever (for ParentChildRetriever)
    if not vector_store and hasattr(retriever, 'child_retriever'):
        child = retriever.child_retriever
        if hasattr(child, 'vector_store'):
            vector_store = child.vector_store
        # Check nested retrievers
        while hasattr(child, 'retrievers') and not vector_store:
            for r in child.retrievers:
                if hasattr(r, 'vector_store'):
                    vector_store = r.vector_store
                    break
            break
    
    if not vector_store:
        if debug:
            print("Could not find vector_store in retriever, skipping neighbor expansion")
        return retrieved_docs
    expander = NeighborExpander(
        vector_store=vector_store,
        max_neighbors=max_neighbors,
        min_overlap_ratio=min_overlap_ratio,
        enable_forward=enable_forward,
        enable_backward=enable_backward,
        debug=debug
    )
    
    return expander.expand(retrieved_docs)