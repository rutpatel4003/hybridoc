from __future__ import annotations
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
from langchain_community.cross_encoders import BaseCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.output_parsers import StrOutputParser
from debug_utils import log_gpu_memory
from typing import Optional, Callable, Tuple
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch

# global singleton to avoid loading embeddings multiple times
_EMBEDDING_MODEL = None

def _stable_doc_uid(doc: Document) -> str:
    """
    Create a stable identifier to dedupe docs across retrievers.
    Prefer metadata-based identity (file/page/chunk) over content hash.
    """
    m = doc.metadata or {}
    source = str(m.get("source", ""))
    file_hash = str(m.get("content_hash", m.get("file_hash", "")))
    page = str(m.get("page", m.get("page_number", "")))
    chunk = str(m.get("chunk_id", m.get("chunk_index", "")))
    ctype = str(m.get("content_type", ""))
    label = str(m.get("label", ""))

    # If we have enough metadata, use that.
    if source or file_hash or page or chunk or label:
        raw = f"{source}|{file_hash}|{page}|{chunk}|{ctype}|{label}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    # Fallback: content hash
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

def reciprocal_rank_fusion(
    results_lists: List[List[Document]],
    k: int = 60,
    boost_fn: Optional[Callable[[Document], float]] = None,
    list_weights: Optional[List[float]] = None,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) with optional per-document boosting and per-list weighting.

    boost_fn(doc) should return a multiplier (e.g., 1.0 normal, 1.5 for tables).
    list_weights: per-retriever weight multipliers (e.g., [1.0, 1.35] to boost table-only results)
    """
    doc_scores: dict[str, float] = {}
    doc_objects: dict[str, Document] = {}

    if list_weights is None:
        list_weights = [1.0] * len(results_lists)

    if len(list_weights) != len(results_lists):
        list_weights = [1.0] * len(results_lists)

    for retriever_idx, results in enumerate(results_lists):
        list_weight = float(list_weights[retriever_idx])
        for rank, doc in enumerate(results):
            doc_id = _stable_doc_uid(doc)
            base = 1.0 / (k + rank + 1)

            mult = 1.0
            if boost_fn is not None:
                try:
                    mult = float(boost_fn(doc))
                    if mult <= 0:
                        mult = 1.0
                except Exception:
                    mult = 1.0

            score = base * mult * list_weight

            if doc_id in doc_scores:
                doc_scores[doc_id] += score
            else:
                doc_scores[doc_id] = score
                doc_objects[doc_id] = doc

    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    fused_docs = [doc_objects[i] for i in sorted_doc_ids]

    if fused_docs:
        top_score = doc_scores[sorted_doc_ids[0]]
        print(f" RRF: Fused {len(fused_docs)} unique docs, top score: {top_score:.3f}")

    return fused_docs

class RRFRetriever(BaseRetriever):
    """
    Custom retriever that uses Reciprocal Rank Fusion (RRF) with optional boosting and list weights.
    """
    retrievers: List[BaseRetriever]
    k: int = 60
    boost_fn: Optional[Callable[[Document], float]] = None
    list_weights: Optional[List[float]] = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results_lists = []
        for retriever in self.retrievers:
            results_lists.append(retriever.invoke(query))

        return reciprocal_rank_fusion(
            results_lists, 
            k=self.k, 
            boost_fn=self.boost_fn, 
            list_weights=self.list_weights
        )


class TablePreservingCompressor:
    """
    Wrap a compressor/reranker and keep at least N table chunks when present.

    Cross-encoders often under-rank linearized tables vs prose.
    This ensures at least a small quota of tables survives compression.
    """

    def __init__(self, base_compressor, min_tables: int = 1):
        self.base_compressor = base_compressor
        self.min_tables = max(0, int(min_tables))

    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        compressed = self.base_compressor.compress_documents(documents, query, callbacks=callbacks)
        if self.min_tables == 0:
            return compressed

        candidate_tables = [d for d in documents if d.metadata.get("content_type") == "table"]
        if not candidate_tables:
            return compressed

        kept_tables = [d for d in compressed if d.metadata.get("content_type") == "table"]
        if len(kept_tables) >= self.min_tables:
            return compressed

        needed = self.min_tables - len(kept_tables)
        compressed_keys = {
            (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("content_hash") or hash(d.page_content))
            for d in compressed
        }

        to_add: List[Document] = []
        for t in candidate_tables:
            key = (t.metadata.get("source"), t.metadata.get("page"), t.metadata.get("content_hash") or hash(t.page_content))
            if key not in compressed_keys:
                to_add.append(t)
                if len(to_add) >= needed:
                    break

        if not to_add:
            return compressed

        out = list(compressed) + to_add

        # trim to base compressor top_n if available
        max_len = getattr(self.base_compressor, "top_n", None)
        if isinstance(max_len, int) and max_len > 0:
            out = out[:max_len]

        return out
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

# LLM for chatbot/query operations
# Stays loaded (keep_alive=-1) for fast query responses
_CHATBOT_LLM = None

def get_chatbot_llm() -> ChatOllama:
    """Get LLM for chatbot/query tasks.
    
    Uses keep_alive=-1 to stay loaded for fast responses.
    Only call this AFTER indexing is complete.
    
    Note: Indexing components (metadata_extractor, table_enricher) 
    create their own short-lived LLMs with keep_alive=0.
    """
    global _CHATBOT_LLM
    if _CHATBOT_LLM is None:
        print("Creating chatbot LLM (will stay loaded)...")
        _CHATBOT_LLM = ChatOllama(
            model=Config.Model.NAME,
            temperature=Config.Model.TEMPERATURE,
            num_ctx=8192,
            num_predict=1024,
            num_thread=8,
            keep_alive=-1,  # stay loaded for queries
            streaming=True,
        )
    return _CHATBOT_LLM

def create_llm() -> ChatOllama:
    """Deprecated: Use get_chatbot_llm() instead."""
    return get_chatbot_llm()

def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
    """
    Generate contextual information for a chunk using LLM.
    This implements Anthropic's contextual retrieval technique.
    """
    try:
        # truncate document if too long
        doc_preview = document[:3000] if len(document) > 3000 else document

        chain = CONTEXT_PROMPT | llm | StrOutputParser()
        context = chain.invoke({
            'document': doc_preview,
            'chunk': chunk[:800]  # limit chunk size for prompt
        })
        # clean thinking tags if present
        if '</think>' in context:
            context = context.split('</think>')[-1].strip()
        return context.strip()

    except Exception as e:
        print(f"    Context generation failed: {e}")
        return ""

class HFSeqClsCrossEncoder(BaseCrossEncoder):
    """
    Pairwise cross-encoder wrapper (query, doc) --> score
    """
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 1024,
        batch_size: int = 8,
        use_fp16: bool = True,
        use_bnb_4bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = "right"
        # make sure padding exists (prevents batch_size>1 crash)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.pad_token is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        model_kwargs = {}
        torch_dtype = None
        can_cuda = (device == "cuda") and torch.cuda.is_available()
        use_fp16 = use_fp16 and can_cuda
        if use_bnb_4bit and can_cuda:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        else:
            # normal fp16/fp32
            torch_dtype = torch.float16 if use_fp16 else torch.float32
            model_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs,
        )
        # if not quantized + device_map=auto, we move explicitly
        if not (use_bnb_4bit and can_cuda):
            self.model = self.model.to("cuda" if can_cuda else "cpu")

        # ensure model also has pad_token_id
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self._use_autocast = use_fp16  # only autocast on CUDA fp16

    @torch.inference_mode()
    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        if not text_pairs:
            return []
        instruction = "Given a query A and a passage B, determine whether the passage contains an answer to the query: "
        device = next(self.model.parameters()).device
        out: List[float] = []

        for i in range(0, len(text_pairs), self.batch_size):
            batch = text_pairs[i : i + self.batch_size]
            queries = [instruction + q for (q, _) in batch]
            docs = [d for (_, d) in batch]

            enc = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            if self._use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**enc).logits
            else:
                logits = self.model(**enc).logits

            # logits shape usually [B, 1] or [B, 2]; handle both
            if logits.dim() == 2 and logits.size(-1) == 1:
                scores = logits.squeeze(-1)
            else:
                # if 2-class, treat "relevant" as last logit
                scores = logits[:, -1]

            out.extend(scores.detach().float().cpu().tolist())

        return out


def create_reranker(
    top_n: int,
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 1024,
    use_bnb_4bit: bool = False,
):
    model_name = "BAAI/bge-reranker-v2-m3"
    actual_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    ce = HFSeqClsCrossEncoder(
        model_name=model_name,
        device=actual_device,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=True,
        use_bnb_4bit=use_bnb_4bit,
    )
    return CrossEncoderReranker(model=ce, top_n=top_n)

def create_embeddings():
    """
    Load embedding model with auto device detection.
    Uses a global singleton to avoid double-loading on GPU.
    Falls back to CPU if GPU memory is low.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    model_name = Config.Preprocessing.EMBEDDING_MODEL

    # check config for device preference
    embedding_device = getattr(Config.Preprocessing, 'EMBEDDING_DEVICE', 'auto')

    if embedding_device == "cpu":
        device = "cpu"
    elif embedding_device == "cuda":
        device = "cuda"
    else:  # "auto" - check GPU memory
        device = Config.DEVICE
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    # Get free memory
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                    free_gb = free_memory / (1024**3)
                    # need at least 2GB free for embeddings + working memory
                    if free_gb < 2.0:
                        print(f"Low GPU memory ({free_gb:.1f}GB free), using CPU for embeddings")
                        device = "cpu"
                    else:
                        print(f"GPU memory available: {free_gb:.1f}GB")
            except Exception as e:
                print(f"Could not check GPU memory: {e}, using CPU")
                device = "cpu"

    print(f"Loading embeddings on device: {device}")
    model_kwargs = {
        "device": device,
        "trust_remote_code": True,
    }
    # add batch size to prevent OOM during embedding
    encode_kwargs = {
        "normalize_embeddings": True,
        "batch_size": getattr(Config.Performance, "EMBEDDING_BATCH_SIZE", 8),
    }
    _EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return _EMBEDDING_MODEL

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
        # get text before this table/figure
        text_before = content[last_end:start].strip()
        # get some context after (look ahead up to 500 chars or next structure)
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
        # create chunk for table/figure with surrounding context
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

_CROSS_PAGE_TAG_OPEN = "[CROSS_PAGE_BRIDGE]"
_CROSS_PAGE_TAG_CLOSE = "[/CROSS_PAGE_BRIDGE]"
def _inject_cross_page_continuations(
    chunks: list[Document],
    chars: int = 600,
    min_text_chars: int = 120,
) -> list[Document]:
    """Inject a short snippet from the next substantial TEXT chunk into the last TEXT chunk of each page.
    Purpose: reduce retrieval failures when a sentence continues on the next page but is interrupted by a table/figure.
    This is deterministic (no punctuation heuristics) and skips non-text and tiny/caption-like text blocks.
    """
    if not chunks or chars <= 0:
        return chunks

    def _page(d: Document):
        m = d.metadata or {}
        return m.get("page", m.get("page_number"))

    def _ctype(d: Document) -> str:
        m = d.metadata or {}
        return str(m.get("content_type", "text"))
    # iterate in original order but use page numbers to find page boundaries.
    out = list(chunks)

    # for each text chunk, check if it's the last text chunk of its page.
    for idx, d in enumerate(out):
        if _ctype(d) != "text":
            continue
        p = _page(d)
        if p is None:
            continue

        # if there's a later TEXT chunk on the same page, this isn't the last.
        is_last_text_on_page = True
        for j in range(idx + 1, len(out)):
            d2 = out[j]
            p2 = _page(d2)
            if p2 is None:
                continue
            if p2 != p:
                break
            if _ctype(d2) == "text" and (d2.page_content or "").strip():
                is_last_text_on_page = False
                break
        if not is_last_text_on_page:
            continue
        # find the next substantial TEXT chunk on a later page.
        continuation_text = ""
        for j in range(idx + 1, len(out)):
            d2 = out[j]
            p2 = _page(d2)
            if p2 is None:
                continue
            if p2 <= p:
                continue
            if _ctype(d2) != "text":
                continue
            cand = (d2.page_content or "").strip()
            if len(cand) < min_text_chars:
                continue
            continuation_text = cand
            break

        if not continuation_text:
            continue

        # don't double-inject
        if _CROSS_PAGE_TAG_OPEN in (d.page_content or ""):
            continue
        snippet = continuation_text[:chars].strip()
        if not snippet:
            continue
        new_content = (d.page_content or "").rstrip() + f"\n\n{_CROSS_PAGE_TAG_OPEN}\n{snippet}\n{_CROSS_PAGE_TAG_CLOSE}"
        out[idx] = Document(page_content=new_content, metadata=d.metadata)
    return out

def _create_chunks_from_blocks(file: File, file_hash: str, llm=None) -> List[Document]:
    """
    Create chunks from structured ContentBlocks, preserving page/type metadata.
    Merges adjacent text blocks and adds context around tables/figures.
    """
    last_text_context = ""
    last_page = None

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
        context_size = getattr(Config.Preprocessing, 'TABLE_FIGURE_CONTEXT_CHARS', 500)
        last_text_context = buffer_text[-context_size:]
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
        context_size = getattr(Config.Preprocessing, 'TABLE_FIGURE_CONTEXT_CHARS', 500)
        context_before = last_text_context[-context_size:] if last_text_context else ""
        context_after = next_text_on_page(i, block.page_num)[:context_size]
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
            # build table content with semantic enrichment
            table_content = block.content
            if block.table_data:
                searchable = block.table_data.to_searchable_text()
                # cap searchable text to keep embeddings sane
                MAX_SEARCHABLE = 5000
                if len(searchable) > MAX_SEARCHABLE:
                    searchable = searchable[:MAX_SEARCHABLE] + "\n[TRUNCATED]"
                table_content += f"\n{searchable}"

            # add semantic description if enabled
            if getattr(Config.Preprocessing, 'ENABLE_TABLE_SEMANTIC_ENRICHMENT', False) and llm:
                print(f"    Enriching table with semantic description...")
                from table_enricher import enrich_table_content
                table_content = enrich_table_content(table_content, llm)

            # include table label in marker for better targeting
            label = block.caption_label or ""
            # put TABLE first so it's unmistakably a table chunk
            content = f"[TABLE:{label}]\n**THIS IS A TABLE - READ ROW BY ROW:**\n{table_content}\n[/TABLE]"
            # keep context, but after table to reduce prefix collisions and improve table retrieval
            if context_before:
                content += f"\n\n[CONTEXT_BEFORE]\n{context_before}\n[/CONTEXT_BEFORE]"
            if context_after:
                content += f"\n\n[CONTEXT_AFTER]\n{context_after}\n[/CONTEXT_AFTER]"
            chunks.append(Document(
                page_content=content,
                metadata={
                    **base_metadata,
                    'table_data': table_data_json,
                    'label': label,
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
    if getattr(Config.Preprocessing, "ENABLE_CROSS_PAGE_BRIDGING", False):
        chunks = _inject_cross_page_continuations(
            chunks,
            chars=int(getattr(Config.Preprocessing, "CROSS_PAGE_CONTEXT_CHARS", 600)),
            min_text_chars=int(getattr(Config.Preprocessing, "CROSS_PAGE_MIN_TEXT_CHARS", 120)),
        )
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
                    print(f"    Enriching table with semantic description...")
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

            print(f"Rebuilt parent store: {len(_parent_store)} parents loaded")
    except Exception as e:
        print(f"Warning: could not rebuild parent store: {e}")

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

        # get children
        children = self.child_retriever.invoke(query)

        # expand to parents
        if AppConfig.Preprocessing.ENABLE_PARENT_CHILD and _parent_store:
            parents = expand_to_parents(children)
            print(f"  Expanded {len(children)} children → {len(parents)} parents")
            return parents

        return children

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Async version"""
        return self._get_relevant_documents(query, **kwargs)

class TableIntentDetector:
    """Domain-agnostic detection of structured/table-seeking queries."""
    _TABLE_REF_RE = re.compile(r"\b(?:table|tab\.?)\s*(\d+|[ivxlcdm]+)\b", re.IGNORECASE)
    _STRUCTURED_INTENT_RE = re.compile(
        r"\b(?:row|rows|column|columns|cell|cells|header|headers|field|fields)\b"
        r"|"
        r"\b(?:in the table|from the table|shown in the table|as shown in the table)\b"
        r"|"
        r"\b(?:spreadsheet|sheet|csv|tsv)\b"
        r"|"
        r"\b(?:values|numbers|figures|breakdown|summary|list|enumerate|compare|comparison|vs|versus)\b"
        r"|"
        r"\b(?:matrix|grid)\b",
        re.IGNORECASE
    )
    @classmethod
    def is_table_query(cls, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        return bool(cls._TABLE_REF_RE.search(q) or cls._STRUCTURED_INTENT_RE.search(q))

class TableSafeCompressionRetriever(BaseRetriever):
    """
    Applies reranking/compression but guarantees table chunks survive for table-intent queries.

    Behavior:
    - Non-table queries: normal ContextualCompressionRetriever behavior.
    - Table-intent queries:
        * Keep up to `keep_tables` table docs from the base retrieval.
        * Rerank ONLY the non-table docs.
        * Merge kept tables + top reranked non-tables, preserving stable dedupe.
    """
    base_retriever: BaseRetriever
    compressor: Any  # reranker compressor
    k_final: int = 12
    keep_tables: int = 3

    model_config = {"arbitrary_types_allowed": True}
    def _dedupe_keep_order(self, docs: List[Document]) -> List[Document]:
        seen = set()
        out = []
        for d in docs:
            uid = _stable_doc_uid(d)
            if uid in seen:
                continue
            seen.add(uid)
            out.append(d)
        return out

    def _is_table_doc(self, doc: Document) -> bool:
        md = doc.metadata or {}
        return str(md.get("content_type", "")).lower() == "table"

    def _compress(self, docs: List[Document], query: str) -> List[Document]:
        # compressor is expected to implement compress_documents(docs, query)
        if hasattr(self.compressor, "compress_documents"):
            return self.compressor.compress_documents(docs, query)
        # fallback: if it's a LangChain-style runnable, try invoke
        if hasattr(self.compressor, "invoke"):
            return self.compressor.invoke({"documents": docs, "query": query})
        return docs

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)

        # if not a table query --> normal rerank/compress on everything
        if not TableIntentDetector.is_table_query(query):
            compressed = self._compress(docs, query)
            return self._dedupe_keep_order(compressed)[: self.k_final]

        # table-intent query --> keep tables, rerank only non-tables
        table_docs = [d for d in docs if self._is_table_doc(d)]
        non_table_docs = [d for d in docs if not self._is_table_doc(d)]
        kept_tables = table_docs[: self.keep_tables]
        reranked_non_tables = self._compress(non_table_docs, query)
        merged = self._dedupe_keep_order(kept_tables + reranked_non_tables)
        # guarantee at least keep_tables tables if available in initial retrieval
        return merged[: self.k_final]

class TableAwareRetriever(BaseRetriever):
    """
    Routes table-seeking queries to a table-only hybrid retriever first,
    then merges with general hybrid results. Also boosts tables when query hints tables.
    """
    base_retriever: BaseRetriever
    table_retriever: BaseRetriever
    k_final: int = 12
    min_table_results: int = 3

    _TABLE_REF_RE = re.compile(
        r"\b(?:table|tab\.?)\s*(\d+|[ivxlcdm]+)\b",
        flags=re.IGNORECASE
    )

    _TABLE_CAPTION_RE = re.compile(r"\bTable\s*(\d+)\b", re.IGNORECASE)
    # medium precision: user asking for structured/tabular fields
    _STRUCTURED_INTENT_RE = re.compile(
        r"\b(?:row|rows|column|columns|cell|cells|header|headers|field|fields)\b"
        r"|"
        r"\b(?:in the table|from the table|shown in the table|as shown in the table)\b"
        r"|"
        r"\b(?:spreadsheet|sheet|csv|tsv)\b"
        r"|"
        r"\b(?:values|numbers|figures|breakdown|summary|list|enumerate|compare|comparison|vs|versus)\b"
        r"|"
        r"\b(?:matrix|grid)\b",
        flags=re.IGNORECASE
    )

    def _inject_tables_from_captions(self, docs: List[Document]) -> List[Document]:
        """
        If retrieved docs mention 'Table N' (often captions in text chunks),
        force-fetch table chunks and prepend them.
        """
        targets = set()
        for d in docs:
            text = (d.page_content or "")[:1500]  # scan a prefix; captions are near the start
            for m in self._TABLE_CAPTION_RE.finditer(text):
                targets.add(m.group(1))

        if not targets:
            return docs

        fetched: List[Document] = []
        for n in sorted(targets):
            # query table retriever explicitly
            fetched.extend(self.table_retriever.invoke(f"Table {n}"))

        # tables first, then original (dedup)
        return self._dedupe_keep_order(fetched + docs)

    def _is_table_query(self, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        # strong signal: explicit "Table X"
        if self._TABLE_REF_RE.search(q):
            return True
        # generic structured intent
        if self._STRUCTURED_INTENT_RE.search(q):
            return True

        return False

    def _prefer_matching_table_number(self, query: str, docs: List[Document]) -> List[Document]:
        """
        If query contains 'Table N', try to bubble matching table label/id to front.
        """
        m = self._TABLE_REF_RE.search(query)
        if not m:
            return docs
        target = m.group(1)

        def key(d: Document) -> Tuple[int, int]:
            md = d.metadata or {}
            ctype = str(md.get("content_type", ""))
            label = str(md.get("caption_label", md.get("label", md.get("table_id", ""))))
            # table + label mentions N
            hit = (ctype.lower() == "table") and (re.search(rf"\b{target}\b", label) is not None)
            # rank: matching tables first, then other tables, then rest
            if hit:
                return (0, 0)
            if ctype.lower() == "table":
                return (1, 0)
            return (2, 0)

        return sorted(docs, key=key)

    def _dedupe_keep_order(self, docs: List[Document]) -> List[Document]:
        seen = set()
        out = []
        for d in docs:
            uid = _stable_doc_uid(d)
            if uid in seen:
                continue
            seen.add(uid)
            out.append(d)
        return out

    def _ensure_min_tables(self, query: str, merged: List[Document], table_docs: List[Document]) -> List[Document]:
        """
        Guarantee at least min_table_results table docs for table intent,
        unless not available.
        """
        want = self.min_table_results
        have = [d for d in merged if (d.metadata or {}).get("content_type") == "table"]
        if len(have) >= want:
            return merged
        # add more table docs at the front (deduped)
        extras = []
        for d in table_docs:
            if (d.metadata or {}).get("content_type") != "table":
                continue
            extras.append(d)

        merged2 = self._dedupe_keep_order(extras + merged)
        return merged2

    def _get_relevant_documents(self, query: str) -> List[Document]:
        table_intent = self._is_table_query(query)

        base_docs = self.base_retriever.invoke(query)

        # caption->table bridge (works even if user didn’t say "Table 2")
        base_docs = self._inject_tables_from_captions(base_docs)

        if not table_intent:
            base_docs = self._prefer_matching_table_number(query, base_docs)
            return base_docs[: self.k_final]

        table_docs = self.table_retriever.invoke(query)

        merged = self._dedupe_keep_order(table_docs + base_docs)
        merged = self._ensure_min_tables(query, merged, table_docs)
        merged = self._prefer_matching_table_number(query, merged)

        return merged[: self.k_final]

class CachedRetriever(BaseRetriever):
    """
    Retriever wrapper that adds semantic query caching.
    """
    base_retriever: BaseRetriever
    embedding_model: any = None
    cache: any = None
    cache_namespace: str = ""
    model_config = {"arbitrary_types_allowed": True}

    def _get_query_embedding(self, query: str):
        if self.embedding_model is None:
            self.embedding_model = create_embeddings()
        return self.embedding_model.embed_query(query)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        from config import Config as AppConfig
        import numpy as np
        if not AppConfig.Performance.ENABLE_QUERY_CACHE:
            return self.base_retriever.invoke(query)

        if self.cache is None:
            from query_cache import get_cache
            self.cache = get_cache()
        cache_key = f"[{self.cache_namespace}] {query}" if self.cache_namespace else query
        query_embedding = np.array(self._get_query_embedding(cache_key))
        cached_docs = self.cache.get(cache_key, query_embedding)
        if cached_docs is not None:
            # if it's table intent but cache has 0 tables, bypass
            try:
                table_intent = TableIntentDetector.is_table_query(query)
            except Exception:
                table_intent = False

            if table_intent:
                has_table = any(
                    str((d.metadata or {}).get("content_type", "")).lower() == "table"
                    for d in cached_docs
                )
                if not has_table:
                    print("  Cache BYPASS: table query but cached docs contain 0 tables")
                else:
                    return cached_docs
            else:
                return cached_docs

        # cache miss (or bypass) → retrieve fresh
        docs = self.base_retriever.invoke(query)
        self.cache.put(cache_key, query_embedding, docs)
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
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        if self.llm is None:
            # use chatbot LLM (query-time, stays loaded)
            self.llm = get_chatbot_llm()

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
            # clean thinking tags
            if '</think>' in result:
                result = result.split('</think>')[-1].strip()
            # parse alternatives
            alternatives = [q.strip() for q in result.strip().split('\n') if q.strip()]
            alternatives = [q for q in alternatives if len(q) > 10][:2]  # Max 2

            return [original_query] + alternatives

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [original_query]

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve with multiple query variations"""
        from config import Config as AppConfig

        if not AppConfig.Chatbot.ENABLE_MULTI_QUERY:
            return self.base_retriever.invoke(query)
        queries = self._generate_queries(query)
        print(f"  Multi-query: {len(queries)} variations")
        # retrieve with all queries
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
    # log initial GPU memory state for debugging
    import torch
    def log_gpu(step: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU [{step}]: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    log_gpu("START of ingest_files")
    # initialize LLM for contextual embeddings + metadata extraction + table enrichment
    llm = None
    needs_llm = (
        Config.Preprocessing.CONTEXTUALIZE_CHUNKS or 
        getattr(Config.Preprocessing, 'ENABLE_METADATA_EXTRACTION', True) or
        getattr(Config.Preprocessing, 'ENABLE_TABLE_SEMANTIC_ENRICHMENT', False)
    )
    if needs_llm:
        print("Initializing LLM for contextual processing...")
        log_gpu("Before create_llm()")
        llm = create_llm()
        log_gpu("After create_llm()")
    # initialize metadata extractor
    use_metadata_extraction = getattr(Config.Preprocessing, 'ENABLE_METADATA_EXTRACTION', True)
    metadata_extractor = None
    if use_metadata_extraction:
        log_gpu("Before metadata_extractor init")
        metadata_extractor = get_metadata_extractor(use_llm=True)
        log_gpu("After metadata_extractor init")
        print("metadata extraction enabled (LLM-powered)")
    # initialize embeddings
    print("📦 Loading embedding model...")
    log_gpu("Before embedding model")
    embedding_model = create_embeddings()
    log_gpu("After embedding model")
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
        existing_sources: dict[str, str] = {}
        if existing_data and 'metadatas' in existing_data:
            for m in existing_data['metadatas']:
                if not m: continue
                source_name = m.get('source')
                if not source_name: continue
                h = m.get('file_hash') or m.get('content_hash')
                if h and source_name not in existing_sources:
                    existing_sources[source_name] = h

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
        # llm is passed to enable table semantic enrichment (improves retrieval)
        if Config.Preprocessing.ENABLE_PARENT_CHILD:
            child_chunks, parent_chunks = _create_parent_child_chunks(f, file_hash, llm=llm)
            _build_parent_store(parent_chunks)
            file_chunks = child_chunks + parent_chunks
            print(f"  created {len(child_chunks)} children, {len(parent_chunks)} parents")
            log_gpu_memory(f"After parent-child chunking {f.name}")
        else:
            log_gpu_memory(f"Before standard chunking {f.name}")
            file_chunks = _create_chunks_from_blocks(f, file_hash, llm=llm)
            log_gpu_memory(f"After standard chunking {f.name}")

        # extract document-level metadata
        if metadata_extractor:
            log_gpu_memory(f"Before document metadata extraction {f.name}")
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
            log_gpu_memory(f"After document metadata extraction {f.name}")

        # enrich each chunk with chunk-level metadata
        if metadata_extractor:
            log_gpu_memory(f"Before chunk-level metadata enrichment {f.name}")
            enriched_chunks = []
            for idx, chunk in enumerate(file_chunks):
                # periodic logging inside loop
                if idx % 10 == 0:
                    log_gpu_memory(f"Processing chunk {idx}/{len(file_chunks)} of {f.name}")
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

        # force cleanup before embedding phase to prevent OOM
        # previous models (metadata extraction LLM) may not have fully released GPU memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_gpu_memory("Before embedding generation (after cleanup)")

        vector_store.add_documents(new_chunks)
        log_gpu_memory("After embedding generation")
    else:
        print('no new content to index')

    # create semantic retriever using the existing vector_store instance
    # reuse the same vector_store to avoid Qdrant lock issues
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
    global_corpus: List[Document] = []
    table_corpus: List[Document] = []

    for t, m in zip(stored_texts, stored_metadatas):
        safe_m = m if m else {}

        # skip parents for BM25
        if Config.Preprocessing.ENABLE_PARENT_CHILD and safe_m.get('is_parent'):
            continue

        doc = Document(page_content=t, metadata=safe_m)
        global_corpus.append(doc)

        if str(safe_m.get("content_type", "")).lower() == "table":
            table_corpus.append(doc)

    print(f'building BM25 index on {len(global_corpus)} chunks')
    bm25_retriever = BM25Retriever.from_documents(global_corpus)
    bm25_retriever.k = Config.Preprocessing.N_BM25_RESULTS

    print(f'building BM25 index on {len(table_corpus)} table chunks')
    bm25_table_retriever = BM25Retriever.from_documents(table_corpus) if table_corpus else None
    if bm25_table_retriever is not None:
        # smaller is better for tables
        bm25_table_retriever.k = max(3, min(8, Config.Preprocessing.N_BM25_RESULTS))
    # semantic (table-only)
    table_filter = {"content_type": "table"}
    if Config.Preprocessing.ENABLE_PARENT_CHILD:
        table_filter = {"is_parent": False, "content_type": "table"}

    semantic_table_retriever = QdrantRetrieverWrapper(
        vector_store=vector_store,
        search_kwargs={
            "k": max(3, min(8, Config.Preprocessing.N_SEMANTIC_RESULTS)),
            "filter": table_filter
        }
    )

    def _boost_tables(doc: Document) -> float:
        md = doc.metadata or {}
        ctype = str(md.get("content_type", "")).lower()
        # boost tables globally inside fusion so they don't lose to captions.
        # increased boost to ensure they win
        return 3.0 if ctype == "table" else 1.0
    # create general ensemble retriever (boosted RRF)
    if Config.Preprocessing.USE_RRF:
        print("using Reciprocal Rank Fusion (RRF) for ensemble (with table boost)")
        ensemble_retriever = RRFRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            k=Config.Preprocessing.RRF_K,
            boost_fn=_boost_tables
        )
    else:
        print("using weighted average for ensemble")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
    # create table-only hybrid retriever (RRF)
    table_retrievers: List[BaseRetriever] = [semantic_table_retriever]
    if bm25_table_retriever is not None:
        table_retrievers.append(bm25_table_retriever)
    table_hybrid = RRFRetriever(
        retrievers=table_retrievers,
        k=max(30, Config.Preprocessing.RRF_K),
        boost_fn=None  # already table-only
    )
    # route table-intent queries to tables-first, then merge
    k_final = getattr(Config.Preprocessing, "N_FINAL_RESULTS", 12)
    min_tables = getattr(Config.Preprocessing, "MIN_TABLE_RESULTS", 3)

    table_aware = TableAwareRetriever(
        base_retriever=ensemble_retriever,
        table_retriever=table_hybrid,
        k_final=k_final,
        min_table_results=min_tables
    )

    # wrap with parent-child if enabled
    if Config.Preprocessing.ENABLE_PARENT_CHILD:
        parent_child_retriever = ParentChildRetriever(child_retriever=table_aware)
        final_retriever = parent_child_retriever
    else:
        final_retriever = table_aware
    
    # stratified table retrieval (general-purpose)
    if getattr(Config, "Tables", None) and Config.Tables.ENABLE_STRATIFIED_TABLE_RETRIEVAL:
        if table_corpus:
            print(f"Stratified retrieval: {len(table_corpus)} table chunks get separate lane")
            
            # Qdrant filter for table-only semantic retrieval
            strat_table_filter = {"content_type": "table"}
            if Config.Preprocessing.ENABLE_PARENT_CHILD:
                strat_table_filter["is_parent"] = False

            strat_semantic_table = QdrantRetrieverWrapper(
                vector_store=vector_store,
                search_kwargs={
                    "k": Config.Tables.N_TABLE_SEMANTIC_RESULTS,
                    "filter": strat_table_filter,
                },
            )
            strat_bm25_table = BM25Retriever.from_documents(table_corpus)
            strat_bm25_table.k = Config.Tables.N_TABLE_BM25_RESULTS
            if Config.Preprocessing.USE_RRF:
                strat_table_retriever = RRFRetriever(
                    retrievers=[strat_semantic_table, strat_bm25_table],
                    k=Config.Preprocessing.RRF_K,
                )
            else:
                strat_table_retriever = EnsembleRetriever(
                    retrievers=[strat_semantic_table, strat_bm25_table],
                    weights=[0.6, 0.4],
                )
            if Config.Preprocessing.ENABLE_PARENT_CHILD:
                strat_table_retriever = ParentChildRetriever(child_retriever=strat_table_retriever)

            # fuse the main retriever with the table-only retriever (weighted RRF)
            final_retriever = RRFRetriever(
                retrievers=[final_retriever, strat_table_retriever],
                k=Config.Preprocessing.RRF_K,
                list_weights=[1.0, Config.Tables.TABLE_LIST_WEIGHT],
            )
            print(f"   ✓ Fused with weight={Config.Tables.TABLE_LIST_WEIGHT} for table lane")

    # add reranker
    print("Loading reranker model...")
    log_gpu_memory("Before create_reranker")
    reranker = create_reranker(top_n=Config.Chatbot.N_CONTEXT_RESULTS)
    if torch.cuda.is_available():
        print(f"GPU memory after reranker: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    if reranker is None:
        retriever_with_reranker = final_retriever
    else:
        # use table-safe wrapper instead of generic compression
        # this guarantees table chunks survive reranking for relevant queries
        retriever_with_reranker = TableSafeCompressionRetriever(
            compressor=reranker,
            base_retriever=final_retriever
        )

    # wrap with semantic caching (outermost layer)
    if Config.Performance.ENABLE_QUERY_CACHE:
        print("enabling semantic query caching")
        ns_raw = "|".join(sorted([f"{f.name}:{len(f.content)}" for f in files]))
        cache_ns = hashlib.md5(ns_raw.encode("utf-8")).hexdigest()[:10]
        cached_retriever = CachedRetriever(
            base_retriever=retriever_with_reranker,
            cache_namespace=cache_ns
        )
        final_result = cached_retriever
    else:
        final_result = retriever_with_reranker

    # source routing for multi-file scenarios
    # only activates when query explicitly mentions a source ("in paper X", "from document Y")
    if len(files) > 1:
        try:
            from source_router import SourceRouterRetriever
            final_result = SourceRouterRetriever(
                base_retriever=final_result,
                vector_store=vector_store
            )
            print(f"Source routing enabled ({len(files)} files) - keyword-based, 0 VRAM overhead")
        except Exception as e:
            print(f"Source routing failed to load ({e}), continuing without it")
    if getattr(Config.Performance, 'CLEAR_GPU_AFTER_INDEXING', True):
        try:
            from pdf_loader import cleanup_ocr_model
            cleanup_ocr_model()
            
            # also clear query cache so we don't serve stale results after re-indexing
            from query_cache import clear_cache
            clear_cache()
        except Exception as e:
            pass  # already cleaned or not loaded

    print("Indexing complete! Retriever ready.")
    if torch.cuda.is_available():
        print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return final_result