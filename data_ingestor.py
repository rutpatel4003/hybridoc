from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
from pdf_loader import File
from vector_store_qdrant import QdrantVectorStore
from metadata_extractor import get_metadata_extractor
import hashlib
import json
import re
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.output_parsers import StrOutputParser

# global singleton to avoid loading embeddings multiple times
_EMBEDDING_MODEL = None

def reciprocal_rank_fusion(
    results_lists: List[List[Document]],
    k: int = 60
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) - fuses multiple ranked lists.
    Where:
        - d = document
        - r = retriever (e.g., semantic, BM25)
        - rank_r(d) = rank of document d in retriever r's results
        - k = constant (typically 60)
    """
    doc_scores: dict[str, float] = {}
    doc_objects: dict[str, Document] = {}
    
    # score each document across all result lists
    for retriever_idx, results in enumerate(results_lists):
        for rank, doc in enumerate(results):
            # use content hash as unique ID (handles same doc from multiple retrievers)
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            rrf_score = 1.0 / (k + rank + 1)
            
            # accumulate scores if doc appears in multiple result lists
            if doc_id in doc_scores:
                doc_scores[doc_id] += rrf_score
            else:
                doc_scores[doc_id] = rrf_score
                doc_objects[doc_id] = doc

    # return documents in fused rank order
    sorted_doc_ids = sorted(
        doc_scores.keys(),
        key=lambda x: doc_scores[x],
        reverse=True
    )
    # return documents in fused rank order
    fused_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids]
    if fused_docs:
        top_score = doc_scores[sorted_doc_ids[0]]
        print(f" RRF: Fused {len(fused_docs)} unique docs, top score: {top_score:.3f}")
    
    return fused_docs

class RRFRetriever(BaseRetriever):
    """
    Custom retriever that uses Reciprocal Rank Fusion.
    """
    retrievers: List[BaseRetriever]
    k: int = 60
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Invoke all retrievers and fuse results with RRF
        """
        # get results from each retriever
        results_lists = []
        for retriever in self.retrievers:
            results = retriever.invoke(query)
            results_lists.append(results)
        
        # fuse with RRF
        fused = reciprocal_rank_fusion(results_lists, k=self.k)
        return fused


CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """
You're an expert in document analysis. Your task is to provide brief, relevant context for a chunk of text from the given documents.

Here is the document:
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
1. Identify the main topic or concept discussed in the chunk.
2. Mention any relevant information or comparisons from the broader document content.
3. If applicable, note how this information relates to the overall theme or purpose of the document.
4. Include any key figures, dates, or percentages that provide important context.
5. Do not use phrases like "This chunk discusses" or "This section provided". Instead directly state the discussion.

Please give a short succint context to situate this chunk within the overall document for the purpose of improving search retrieval of the chunk.

Context:
""".strip()
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = Config.Preprocessing.CHUNK_SIZE,
    chunk_overlap = Config.Preprocessing.CHUNK_OVERLAP
)

def create_llm() -> ChatOllama:
    """Create LLM for context generation"""
    return ChatOllama(
        model=Config.Model.NAME,
        temperature=0,
        num_ctx=4096,
        num_predict=256,
        keep_alive=-1,
    )

def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
    """
    Generate contextual information for a chunk using LLM.
    This implements Anthropic's contextual retrieval technique.

    Args:
        llm: Language model for generation
        document: Full document text (or large portion)
        chunk: Specific chunk to contextualize

    Returns:
        2-3 sentence context describing the chunk's role in document
    """
    try:
        # Truncate document if too long
        doc_preview = document[:3000] if len(document) > 3000 else document

        chain = CONTEXT_PROMPT | llm | StrOutputParser()
        context = chain.invoke({
            'document': doc_preview,
            'chunk': chunk[:800]  # Limit chunk size for prompt
        })

        # Clean thinking tags if present
        if '</think>' in context:
            context = context.split('</think>')[-1].strip()

        return context.strip()

    except Exception as e:
        print(f"    Context generation failed: {e}")
        return ""

def create_reranker():
    """
    Create a cross-encoder reranker with auto device detection.
    """
    model_name = Config.Preprocessing.RERANKER
    device = Config.DEVICE
    
    print(f"Loading reranker on device: {device}")
    
    model_kwargs = {'device': device}
    
    model = HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    
    return CrossEncoderReranker(model=model, top_n=Config.Chatbot.N_CONTEXT_RESULTS)

def create_embeddings():
    """
    Load embedding model with auto device detection.
    Uses a global singleton to avoid double-loading on GPU.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    model_name = Config.Preprocessing.EMBEDDING_MODEL
    device = Config.DEVICE

    print(f"Loading embeddings on device: {device}")

    model_kwargs = {
        "device": device,
        "trust_remote_code": True,
    }
    encode_kwargs = {"normalize_embeddings": True}

    _EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return _EMBEDDING_MODEL

# def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
#     messages = CONTEXT_PROMPT.format_messages(document=document, chunk=chunk)
#     response = llm.invoke(messages)
#     return response.content

def _detect_content_type(text: str) -> str:
    """
    Content type detection based on markers
    """
    if "[TABLE:" in text and "[/TABLE]" in text:
        return 'table'
    if "[FIGURE]" in text and "[/FIGURE]" in text:
        return 'figure'
    return 'text'

def _create_chunks(document: Document) -> List[Document]:
    """
    Create chunks - tables/figures get surrounding context, text gets split normally.
    FIXED: Tables/figures include nearby text for better retrieval.
    """
    content = document.page_content
    
    # pattern to find tables and figures
    pattern = r'(\[TABLE:.*?\[/TABLE\]|\[FIGURE\].*?\[/FIGURE\])'
    
    # find all structured content with their positions
    structured_matches = list(re.finditer(pattern, content, flags=re.DOTALL))
    
    if not structured_matches:
        # no tables/figures - just split normally
        chunks = text_splitter.create_documents(
            [content],
            metadatas=[{**document.metadata, 'content_type': 'text'}]
        )
        return chunks
    
    final_chunks = []
    last_end = 0
    
    for match in structured_matches:
        start, end = match.start(), match.end()
        structured_content = match.group()
        
        # get text BEFORE this table/figure
        text_before = content[last_end:start].strip()
        
        # get some context AFTER (look ahead up to 500 chars or next structure)
        context_after_end = min(end + 500, len(content))
        next_match = re.search(pattern, content[end:context_after_end], flags=re.DOTALL)
        if next_match:
            context_after_end = end + next_match.start()
        text_after = content[end:context_after_end].strip()
        
        # chunk the text before (if substantial)
        if text_before and len(text_before) > 50:
            text_chunks = text_splitter.create_documents(
                [text_before],
                metadatas=[{**document.metadata, 'content_type': 'text'}]
            )
            final_chunks.extend(text_chunks)
        
        # determine content type
        content_type = 'table' if '[TABLE:' in structured_content else 'figure'
        
        # create chunk for table/figure WITH surrounding context
        # include last 200 chars of text_before + table + first 200 chars of text_after
        context_before = text_before[-200:] if len(text_before) > 200 else text_before
        context_after_snippet = text_after[:200] if len(text_after) > 200 else text_after
        
        chunk_with_context = ""
        if context_before:
            chunk_with_context += f"[CONTEXT]\n{context_before}\n[/CONTEXT]\n\n"
        chunk_with_context += structured_content
        if context_after_snippet:
            chunk_with_context += f"\n\n[CONTEXT]\n{context_after_snippet}\n[/CONTEXT]"
        
        final_chunks.append(Document(
            page_content=chunk_with_context,
            metadata={**document.metadata, 'content_type': content_type}
        ))
        
        last_end = end
    
    # handle remaining text after last table/figure
    remaining_text = content[last_end:].strip()
    if remaining_text and len(remaining_text) > 50:
        text_chunks = text_splitter.create_documents(
            [remaining_text],
            metadatas=[{**document.metadata, 'content_type': 'text'}]
        )
        final_chunks.extend(text_chunks)
    
    return final_chunks

def _calculate_file_hash(content: str) -> str:
    """
    Calculate hash of file content for deduplication
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def _delete_chunks_by_source(vector_store: QdrantVectorStore, source_name: str):
    """
    Delete all chunks that belong to a specific source file
    called when a file's content hash has changed (needs re-indexing)
    """
    try:
        results = vector_store.get(filter={"source": source_name})
        ids_to_delete = results.get('ids', [])

        if ids_to_delete:
            vector_store.delete(ids_to_delete)
            print(f"deleted {len(ids_to_delete)} stale chunks for '{source_name}'")
    except Exception as e:
        print(f"warning: could not delete old chunks for {source_name}: {e}")

def _create_chunks_from_blocks(file: File, file_hash: str, llm=None) -> List[Document]:
    """
    Create chunks from structured ContentBlocks, preserving page/type metadata.
    Merges adjacent text blocks and adds context around tables/figures.
    """
    last_text_context = ""
    last_page = None

    def _find_next_text(start_idx: int, page_num: int) -> str:
        for j in range(start_idx + 1, len(file.content_blocks)):
            b = file.content_blocks[j]
            if b.page_num != page_num:
                break
            if b.content_type == "text" and b.content.strip():
                return b.content.strip()
        return ""
    if not file.content_blocks:
        doc = Document(
            file.content,
            metadata={'source': file.name, 'content_hash': file_hash}
        )
        return _create_chunks(doc)

    blocks = file.content_blocks
    chunks = []
    full_document = file.content if Config.Preprocessing.CONTEXTUALIZE_CHUNKS else None

    buffer_text = ""
    buffer_page = None
    last_text_context = ""

    def flush_buffer():
        nonlocal buffer_text, buffer_page, last_text_context
        if not buffer_text.strip():
            buffer_text = ""
            buffer_page = None
            return

        # save tail for table context
        last_text_context = buffer_text[-400:]

        base_metadata = {
            'source': file.name,
            'content_hash': file_hash,
            'page': buffer_page,
            'content_type': 'text',
        }

        # split if too long
        if len(buffer_text) > Config.Preprocessing.CHUNK_SIZE:
            sub_chunks = text_splitter.create_documents(
                [buffer_text],
                metadatas=[base_metadata]
            )
        else:
            sub_chunks = [Document(page_content=buffer_text, metadata=base_metadata)]

        # contextualize if enabled
        if Config.Preprocessing.CONTEXTUALIZE_CHUNKS and llm and full_document:
            for sc in sub_chunks:
                context = _generate_context(llm, full_document, sc.page_content)
                if context:
                    sc.page_content = f"{context}\n\n{sc.page_content}"
                    sc.metadata['contextualized'] = True

        chunks.extend(sub_chunks)

        buffer_text = ""
        buffer_page = None

    def next_text_on_page(start_idx: int, page_num: int) -> str:
        for j in range(start_idx + 1, len(blocks)):
            b = blocks[j]
            if b.page_num != page_num:
                break
            if b.content_type == 'text' and b.content.strip():
                return b.content.strip()
        return ""

    for i, block in enumerate(blocks):
        if block.content_type == 'text':
            if buffer_page is None:
                buffer_page = block.page_num
            if block.page_num != buffer_page:
                flush_buffer()
                buffer_page = block.page_num

            text = block.content.strip()
            if text:
                buffer_text = (buffer_text + "\n" + text).strip()
            if len(buffer_text) > Config.Preprocessing.CHUNK_SIZE:
                flush_buffer()
            continue

        # table/figure: flush text first
        flush_buffer()

        base_metadata = {
            'source': file.name,
            'content_hash': file_hash,
            'page': block.page_num,
            'content_type': block.content_type,
        }

        context_before = last_text_context[-400:] if last_text_context else ""
        context_after = next_text_on_page(i, block.page_num)[:400]

        if block.content_type == 'table':
            table_data_json = None
            if block.table_data:
                table_dict = block.table_data.to_dict()
                table_data_json = json.dumps(table_dict)
                MAX_TABLE_DATA_SIZE = 30000
                if len(table_data_json) > MAX_TABLE_DATA_SIZE:
                    truncated = {
                        'headers': table_dict['headers'],
                        'rows': table_dict['rows'][:10],
                        'raw_markdown': table_dict.get('raw_markdown', '')[:5000],
                        'num_rows': table_dict['num_rows'],
                        'num_cols': table_dict['num_cols'],
                        'truncated': True,
                        'rows_shown': min(10, len(table_dict['rows']))
                    }
                    table_data_json = json.dumps(truncated)

            # Build table content with semantic enrichment
            table_content = block.content
            if block.table_data:
                table_content += f"\n{block.table_data.to_searchable_text()}"

            # Add semantic description if enabled
            if getattr(Config.Preprocessing, 'ENABLE_TABLE_SEMANTIC_ENRICHMENT', False) and llm:
                from table_enricher import enrich_table_content
                table_content = enrich_table_content(table_content, llm)

            content = f"[TABLE:]\n{table_content}\n[/TABLE]"

            if context_before:
                content = f"[CONTEXT]\n{context_before}\n[/CONTEXT]\n\n{content}"
            if context_after:
                content = f"{content}\n\n[CONTEXT]\n{context_after}\n[/CONTEXT]"

            chunks.append(Document(
                page_content=content,
                metadata={
                    **base_metadata,
                    'table_data': table_data_json,
                }
            ))

        elif block.content_type == 'figure':
            content = f"[FIGURE]\n{block.content}\n[/FIGURE]"
            if context_before:
                content = f"[CONTEXT]\n{context_before}\n[/CONTEXT]\n\n{content}"
            if context_after:
                content = f"{content}\n\n[CONTEXT]\n{context_after}\n[/CONTEXT]"

            chunks.append(Document(
                page_content=content,
                metadata=base_metadata
            ))

    flush_buffer()
    return chunks

def _create_parent_child_chunks(file: File, file_hash: str, llm=None) -> tuple[List[Document], List[Document]]:
    """
    Create parent-child chunk pairs with table/figure context.
    Parent chunks include surrounding text context so tables are interpretable.
    """
    import uuid

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.Preprocessing.PARENT_CHUNK_SIZE,
        chunk_overlap=200
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.Preprocessing.CHILD_CHUNK_SIZE,
        chunk_overlap=50
    )

    parent_chunks = []
    child_chunks = []
    full_document = file.content if Config.Preprocessing.CONTEXTUALIZE_CHUNKS else None

    # fallback
    if not file.content_blocks:
        parent_docs = parent_splitter.create_documents(
            [file.content],
            metadatas=[{'source': file.name, 'content_hash': file_hash, 'content_type': 'text'}]
        )
    else:
        blocks = file.content_blocks
        parent_docs = []

        last_text_context = ""
        last_page = None

        def update_context(text: str, page_num: int):
            nonlocal last_text_context, last_page
            if last_page != page_num:
                last_text_context = ""
                last_page = page_num
            if text:
                last_text_context = (last_text_context + "\n" + text).strip()
                if len(last_text_context) > 800:
                    last_text_context = last_text_context[-800:]

        def next_text_on_page(start_idx: int, page_num: int) -> str:
            for j in range(start_idx + 1, len(blocks)):
                b = blocks[j]
                if b.page_num != page_num:
                    break
                if b.content_type == "text" and b.content.strip():
                    return b.content.strip()
            return ""
        for i, block in enumerate(blocks):
            base_meta = {
                'source': file.name,
                'content_hash': file_hash,
                'page': block.page_num,
                'content_type': block.content_type,
            }
            # text blocks → parents
            if block.content_type == "text":
                text = block.content.strip()
                update_context(text, block.page_num)

                if len(text) > Config.Preprocessing.PARENT_CHUNK_SIZE:
                    splits = parent_splitter.create_documents(
                        [text],
                        metadatas=[base_meta]
                    )
                    parent_docs.extend(splits)
                elif text:
                    parent_docs.append(Document(page_content=text, metadata=base_meta))
                continue
            # table/figure: wrap with context
            context_before = last_text_context[-400:] if last_text_context else ""
            context_after = next_text_on_page(i, block.page_num)[:400]
            if block.content_type == "table":
                table_content = block.content
                if block.table_data:
                    base_meta['table_data'] = json.dumps(block.table_data.to_dict())[:30000]
                    table_content += f"\n{block.table_data.to_searchable_text()}"

                # Add semantic enrichment if enabled
                if getattr(Config.Preprocessing, 'ENABLE_TABLE_SEMANTIC_ENRICHMENT', False) and llm:
                    from table_enricher import enrich_table_content
                    table_content = enrich_table_content(table_content, llm)

                content = f"[TABLE:]\n{table_content}\n[/TABLE]"

            else:  # figure
                content = f"[FIGURE]\n{block.content}\n[/FIGURE]"

            if context_before:
                content = f"[CONTEXT]\n{context_before}\n[/CONTEXT]\n\n{content}"
            if context_after:
                content = f"{content}\n\n[CONTEXT]\n{context_after}\n[/CONTEXT]"

            parent_docs.append(Document(page_content=content, metadata=base_meta))
    # create children from parents
    for parent in parent_docs:
        parent_id = str(uuid.uuid4())[:8]
        parent.metadata['parent_id'] = parent_id
        parent.metadata['is_parent'] = True
        parent_chunks.append(parent)
        if parent.metadata.get('content_type') in ('table', 'figure'):
            child = Document(
                page_content=parent.page_content,
                metadata={**parent.metadata, 'parent_id': parent_id, 'is_parent': False}
            )
            child_chunks.append(child)
            continue
        if len(parent.page_content) > Config.Preprocessing.CHILD_CHUNK_SIZE:
            children = child_splitter.create_documents(
                [parent.page_content],
                metadatas=[{
                    **parent.metadata,
                    'parent_id': parent_id,
                    'is_parent': False,
                }]
            )
            if Config.Preprocessing.CONTEXTUALIZE_CHUNKS and llm and full_document:
                for child in children:
                    context = _generate_context(llm, full_document, child.page_content)
                    if context:
                        child.page_content = f"{context}\n\n{child.page_content}"
                        child.metadata['contextualized'] = True
            child_chunks.extend(children)
        else:
            child_content = parent.page_content
            if Config.Preprocessing.CONTEXTUALIZE_CHUNKS and llm and full_document:
                context = _generate_context(llm, full_document, child_content)
                if context:
                    child_content = f"{context}\n\n{child_content}"
            child_chunks.append(Document(
                page_content=child_content,
                metadata={**parent.metadata, 'parent_id': parent_id, 'is_parent': False}
            ))

    return child_chunks, parent_chunks

# global parent store (maps parent_id -> parent Document)
_parent_store: dict[str, Document] = {}


def _build_parent_store(parent_chunks: List[Document]):
    """Store parents in memory for lookup during retrieval"""
    global _parent_store
    for parent in parent_chunks:
        pid = parent.metadata.get('parent_id')
        if pid:
            _parent_store[pid] = parent


def _rebuild_parent_store_from_qdrant(vector_store: QdrantVectorStore):
    """
    Rebuild parent store from qdrant on startup
    called when loading existing indexed files
    """
    global _parent_store

    try:
        # get all parent documents from qdrant
        results = vector_store.get(filter={"is_parent": True})

        if results and results.get('documents'):
            docs = results['documents']
            metadatas = results.get('metadatas', [])

            for doc_text, meta in zip(docs, metadatas):
                if meta and meta.get('parent_id'):
                    parent_doc = Document(page_content=doc_text, metadata=meta)
                    _parent_store[meta['parent_id']] = parent_doc

            print(f"rebuilt parent store: {len(_parent_store)} parents loaded")
    except Exception as e:
        print(f"warning: could not rebuild parent store: {e}")


def expand_to_parents(child_docs: List[Document]) -> List[Document]:
    """
    Given retrieved child documents, expand to their parent documents.
    Deduplicates by parent_id.
    """
    global _parent_store

    seen_parents = set()
    expanded = []

    for child in child_docs:
        parent_id = child.metadata.get('parent_id')

        if parent_id and parent_id not in seen_parents:
            parent = _parent_store.get(parent_id)
            if parent:
                expanded.append(parent)
                seen_parents.add(parent_id)
            else:
                # parent not found, use child as-is
                expanded.append(child)
                seen_parents.add(parent_id)
        elif not parent_id:
            # no parent_id, use as-is
            expanded.append(child)

    return expanded


class ParentChildRetriever(BaseRetriever):
    """
    Custom retriever that retrieves children then expands to parents.
    """
    child_retriever: BaseRetriever

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve children, expand to parents"""
        from config import Config as AppConfig

        # Get children
        children = self.child_retriever.invoke(query)

        # Expand to parents
        if AppConfig.Preprocessing.ENABLE_PARENT_CHILD and _parent_store:
            parents = expand_to_parents(children)
            print(f"  Expanded {len(children)} children → {len(parents)} parents")
            return parents

        return children

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Async version"""
        return self._get_relevant_documents(query, **kwargs)


class CachedRetriever(BaseRetriever):
    """
    Retriever wrapper that adds semantic query caching.
    """
    base_retriever: BaseRetriever
    embedding_model: any = None
    cache: any = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_query_embedding(self, query: str):
        """Get embedding for query (used as cache key)"""
        if self.embedding_model is None:
            self.embedding_model = create_embeddings()
        return self.embedding_model.embed_query(query)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve with caching"""
        from config import Config as AppConfig
        import numpy as np

        if not AppConfig.Performance.ENABLE_QUERY_CACHE:
            return self.base_retriever.invoke(query)

        # Initialize cache on first use
        if self.cache is None:
            from query_cache import get_cache
            self.cache = get_cache()

        # Get query embedding for cache lookup
        query_embedding = np.array(self._get_query_embedding(query))

        # Try cache first
        cached_docs = self.cache.get(query, query_embedding)
        if cached_docs is not None:
            return cached_docs

        # Cache miss - retrieve and store
        docs = self.base_retriever.invoke(query)
        self.cache.put(query, query_embedding, docs)

        return docs

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


class MultiQueryRetriever(BaseRetriever):
    """
    Generates multiple query variations and retrieves with all of them.
    """
    base_retriever: BaseRetriever
    llm: any = None

    model_config = {"arbitrary_types_allowed": True}

    def _generate_queries(self, original_query: str) -> List[str]:
        """Generate 2-3 query variations using LLM"""
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        if self.llm is None:
            self.llm = ChatOllama(
                model=Config.Model.NAME,
                temperature=0.3,
                num_ctx=1024,
                num_predict=150,
                keep_alive=-1,
            )

        prompt = ChatPromptTemplate.from_template(
            """Generate 2 alternative versions of this question for better search.
Keep the same meaning but use different words/phrasing.
Output ONLY the 2 alternatives, one per line. No numbering, no explanations.

Original: {query}

Alternatives:"""
        )

        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({'query': original_query})

            # Clean thinking tags
            if '</think>' in result:
                result = result.split('</think>')[-1].strip()

            # Parse alternatives
            alternatives = [q.strip() for q in result.strip().split('\n') if q.strip()]
            alternatives = [q for q in alternatives if len(q) > 10][:2]  # Max 2

            return [original_query] + alternatives

        except Exception as e:
            print(f"  Query expansion failed: {e}")
            return [original_query]

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve with multiple query variations"""
        from config import Config as AppConfig

        if not AppConfig.Chatbot.ENABLE_MULTI_QUERY:
            return self.base_retriever.invoke(query)

        queries = self._generate_queries(query)
        print(f"  Multi-query: {len(queries)} variations")

        # Retrieve with all queries
        all_docs = []
        seen_content = set()

        for q in queries:
            docs = self.base_retriever.invoke(q)
            for doc in docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(content_hash)

        return all_docs

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)

def ingest_files(files: List[File]) -> BaseRetriever:
    """
    Ingest files into Qdrant vector database with rich metadata

    Features:
    - qdrant vector store (scalable)
    - rich metadata extraction (document + chunk level)
    - table intelligence
    - contextual embeddings (optional)
    - semantic caching
    """
    # initialize LLM for contextual embeddings + metadata extraction
    llm = None
    if Config.Preprocessing.CONTEXTUALIZE_CHUNKS or getattr(Config.Preprocessing, 'ENABLE_METADATA_EXTRACTION', True):
        print("initializing LLM for contextual processing...")
        llm = create_llm()

    # initialize metadata extractor
    use_metadata_extraction = getattr(Config.Preprocessing, 'ENABLE_METADATA_EXTRACTION', True)
    metadata_extractor = None
    if use_metadata_extraction:
        metadata_extractor = get_metadata_extractor(use_llm=True)
        print("metadata extraction enabled (LLM-powered)")

    # initialize embeddings
    embedding_model = create_embeddings()

    # connect to qdrant vector store
    vector_store = QdrantVectorStore(
        collection_name='private-rag',
        embedding_function=embedding_model,
        path=str(Config.Path.VECTOR_DB_DIR),
        embedding_dim=768,  # gte-multilingual-base
    )

    # check for existing files (deduplication)
    try:
        existing_data = vector_store.get()
        existing_sources = {}
        if existing_data and 'metadatas' in existing_data:
            for m in existing_data['metadatas']:
                if m and 'source' in m:
                    source_name = m['source']
                    file_hash = m.get('content_hash', None)
                    existing_sources[source_name] = file_hash

        table_count = sum(1 for m in existing_data.get('metadatas', [])
                         if m and m.get('content_type') == 'table')

        print(f'found {len(existing_sources)} files in database ({table_count} table chunks)')
        print(f'files: {list(existing_sources.keys())}')
    except Exception as e:
        print(f'error reading database: {e}')
        existing_sources = {}

    # process files
    new_chunks = []
    skipped_files = []
    document_metadata_store = {}  # store doc-level metadata

    for f in files:
        file_hash = _calculate_file_hash(f.content)

        # check if file already indexed
        if f.name in existing_sources:
            stored_hash = existing_sources[f.name]
            if stored_hash == file_hash:
                print(f'skipping {f.name} (already indexed)')
                skipped_files.append(f.name)
                continue
            else:
                print(f'file {f.name} changed - deleting old chunks and reprocessing')
                _delete_chunks_by_source(vector_store, f.name)

        # process new file
        print(f"indexing: {f.name}")

        # create chunks (with or without parent-child)
        if Config.Preprocessing.ENABLE_PARENT_CHILD:
            child_chunks, parent_chunks = _create_parent_child_chunks(f, file_hash, llm)
            _build_parent_store(parent_chunks)
            file_chunks = child_chunks + parent_chunks
            print(f"  created {len(child_chunks)} children, {len(parent_chunks)} parents")
        else:
            file_chunks = _create_chunks_from_blocks(f, file_hash, llm)

        # extract document-level metadata
        if metadata_extractor:
            doc_metadata = metadata_extractor.extract_document_metadata(
                full_text=f.content,
                filename=f.name,
                chunks=file_chunks
            )
            document_metadata_store[f.name] = doc_metadata.to_dict()

            print(f"  detected: {doc_metadata.doc_type}, {doc_metadata.total_pages} pages, "
                  f"{doc_metadata.language} language")

            # add document-level metadata to all chunks
            for chunk in file_chunks:
                chunk.metadata.update({
                    'doc_title': doc_metadata.title,
                    'doc_type': doc_metadata.doc_type,
                    'doc_language': doc_metadata.language,
                    'doc_total_pages': doc_metadata.total_pages,
                })

        # enrich each chunk with chunk-level metadata
        if metadata_extractor:
            enriched_chunks = []
            for idx, chunk in enumerate(file_chunks):
                enriched_chunk = metadata_extractor.enrich_chunk_metadata(
                    chunk=chunk,
                    chunk_index=idx,
                    full_document=f.content if use_metadata_extraction else None
                )
                enriched_chunks.append(enriched_chunk)
            file_chunks = enriched_chunks

        # count content types
        table_chunks = sum(1 for c in file_chunks if c.metadata.get('content_type') == 'table')
        if table_chunks > 0:
            print(f"  found {table_chunks} table(s)")

        new_chunks.extend(file_chunks)

    # handle skipped files
    if skipped_files:
        print(f'loaded {len(skipped_files)} file(s) from cache')
        if Config.Preprocessing.ENABLE_PARENT_CHILD:
            _rebuild_parent_store_from_qdrant(vector_store)

    # add new chunks to database
    if new_chunks:
        print(f'adding {len(new_chunks)} new chunks to vector database')

        # count content types
        tables = sum(1 for c in new_chunks if c.metadata.get('content_type') == 'table')
        figures = sum(1 for c in new_chunks if c.metadata.get('content_type') == 'figure')
        text = len(new_chunks) - tables - figures

        print(f'{text} text chunks, {tables} tables, {figures} figures')

        vector_store.add_documents(new_chunks)
    else:
        print('no new content to index')

    # create semantic retriever using the existing vector_store instance
    # IMPORTANT: reuse the same vector_store to avoid Qdrant lock issues
    class QdrantRetrieverWrapper(BaseRetriever):
        """wrapper around existing vector store instance"""
        vector_store: Any
        search_kwargs: Dict

        model_config = {"arbitrary_types_allowed": True}

        def _get_relevant_documents(self, query: str) -> List[Document]:
            k = self.search_kwargs.get('k', 4)
            filter_dict = self.search_kwargs.get('filter', None)
            return self.vector_store.similarity_search(query, k=k, filter=filter_dict)

        async def _aget_relevant_documents(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query)

    if Config.Preprocessing.ENABLE_PARENT_CHILD:
        semantic_retriever = QdrantRetrieverWrapper(
            vector_store=vector_store,
            search_kwargs={
                'k': Config.Preprocessing.N_SEMANTIC_RESULTS,
                'filter': {'is_parent': False}
            }
        )
    else:
        semantic_retriever = QdrantRetrieverWrapper(
            vector_store=vector_store,
            search_kwargs={'k': Config.Preprocessing.N_SEMANTIC_RESULTS}
        )

    # create BM25 retriever
    db_state = vector_store.get()
    stored_texts = db_state.get('documents', [])
    stored_metadatas = db_state.get('metadatas', [])

    if not stored_texts:
        raise ValueError('database is empty! please upload a document.')

    # reconstruct documents for BM25 (children only if parent-child enabled)
    global_corpus = []
    for t, m in zip(stored_texts, stored_metadatas):
        safe_m = m if m else {}
        # skip parents for BM25
        if Config.Preprocessing.ENABLE_PARENT_CHILD and safe_m.get('is_parent'):
            continue
        global_corpus.append(Document(page_content=t, metadata=safe_m))

    print(f'building BM25 index on {len(global_corpus)} chunks')
    bm25_retriever = BM25Retriever.from_documents(global_corpus)
    bm25_retriever.k = Config.Preprocessing.N_BM25_RESULTS

    # create ensemble retriever
    if Config.Preprocessing.USE_RRF:
        print("using Reciprocal Rank Fusion (RRF) for ensemble")
        ensemble_retriever = RRFRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            k=Config.Preprocessing.RRF_K
        )
    else:
        print("using weighted average for ensemble")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )

    # wrap with parent-child if enabled
    if Config.Preprocessing.ENABLE_PARENT_CHILD:
        parent_child_retriever = ParentChildRetriever(child_retriever=ensemble_retriever)
        final_retriever = parent_child_retriever
    else:
        final_retriever = ensemble_retriever

    # add reranker
    reranker = create_reranker()
    if reranker is None:
        retriever_with_reranker = final_retriever
    else:
        retriever_with_reranker = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=final_retriever
        )

    # wrap with semantic caching (outermost layer)
    if Config.Performance.ENABLE_QUERY_CACHE:
        print("enabling semantic query caching")
        cached_retriever = CachedRetriever(
            base_retriever=retriever_with_reranker,
            embedding_model=embedding_model
        )
        return cached_retriever

    return retriever_with_reranker