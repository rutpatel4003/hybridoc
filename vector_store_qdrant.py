from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from config import Config
import hashlib

class QdrantVectorStore:
    """
    Qdrant vector store with metadata support 
    """
    def __init__(
        self,
        collection_name: str,
        embedding_function: Any,
        path: str = None,
        url: str = None,
        embedding_dim: int = 768,
    ):
        """Initialize qdrant vector store"""
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim

        # initialize client
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

        # create collection if doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"created qdrant collection: {self.collection_name}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to collection with batched embedding generation"""
        if not documents:
            return []

        # extract texts and metadatas
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # generate embeddings in batches to avoid GPU OOM
        from config import Config
        import torch
        import gc
        batch_size = getattr(Config.Performance, 'EMBEDDING_BATCH_SIZE', 8)
        all_embeddings = []
        print(f"  Generating embeddings in batches of {batch_size} (total: {len(texts)} chunks)")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_function.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # progress indicator
            if (i + batch_size) % (batch_size * 4) == 0 or i + batch_size >= len(texts):
                print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
            
            # cleanup between batches to reduce memory fragmentation
            if i + batch_size < len(texts):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        embeddings = all_embeddings

        # create points
        points = []
        point_ids = []

        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            # generate unique ID (use content hash to create valid UUID)
            chunk_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            # create valid UUID from chunk_hash
            uuid_str = chunk_hash[:32]
            point_id = f"{uuid_str[:8]}-{uuid_str[8:12]}-{uuid_str[12:16]}-{uuid_str[16:20]}-{uuid_str[20:32]}"

            # preserve FILE-level content_hash from metadata (set by data_ingestor)
            file_hash = None
            if isinstance(metadata, dict):
                file_hash = metadata.get("content_hash")

            payload = {
                **(metadata or {}),
                "text": text,
                "chunk_hash": chunk_hash,   # store chunk hash separately
            }

            # if metadata did not have file hash, store one
            # but ideally data_ingestor always provides it
            if file_hash is None:
                payload["content_hash"] = None
            # convert complex metadata to JSON strings
            payload = self._serialize_payload(payload)

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
            points.append(point)
            point_ids.append(point_id)
        # upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return point_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        """Semantic similarity search"""
        # generate query embedding
        query_embedding = self.embedding_function.embed_query(query)

        # build filter if provided
        qdrant_filter = self._build_filter(filter) if filter else None
        # search (version-safe)
        try:
            # query_points returns (points, next_offset)
            if hasattr(self.client, "query_points"):
                result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=k,
                    query_filter=qdrant_filter,
                )
                # unpack tuple
                if isinstance(result, tuple):
                    points, _ = result
                else:
                    points = result
            # fallback: search_points
            elif hasattr(self.client, "search_points"):
                result = self.client.search_points(
                    collection_name=self.collection_name,
                    vector=query_embedding,
                    limit=k,
                    filter=qdrant_filter,
                )
                points = result
            # fallback: search
            else:
                points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=k,
                    query_filter=qdrant_filter,
                )
        except Exception as e:
            print(f"Qdrant search failed: {e}")
            return []

        # convert to documents
        documents = []
        for point in points:
            payload = dict(point.payload) if hasattr(point, 'payload') else point
            
            # safe extraction
            if isinstance(payload, dict):
                text = payload.pop("text", "")
                payload.pop("chunk_hash", None)
                payload = self._deserialize_payload(payload)
                doc = Document(
                    page_content=text,
                    metadata=payload,
                )
                documents.append(doc)

        return documents

    def get(self, filter: Optional[Dict] = None) -> Dict:
        """Get all points matching filter"""
        # build filter
        qdrant_filter = self._build_filter(filter) if filter else None
        # scroll to get all points
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            
            # unpack tuple
            if isinstance(result, tuple):
                points, _ = result
            else:
                points = result
        except Exception as e:
            print(f"Qdrant scroll failed: {e}")
            return {'documents': [], 'metadatas': [], 'ids': []}

        # extract data
        documents = []
        metadatas = []
        ids = []

        for point in points:
            payload = dict(point.payload) if hasattr(point, 'payload') else {}

            text = payload.pop("text", "")
            payload.pop("chunk_hash", None)
            # deserialize complex metadata
            payload = self._deserialize_payload(payload)
            documents.append(text)
            metadatas.append(payload)
            ids.append(str(point.id))
        return {
            'documents': documents,
            'metadatas': metadatas,
            'ids': ids,
        }

    def delete(self, ids: List[str]):
        """Delete points by ID"""
        if not ids:
            return

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
            )
        except Exception as e:
            print(f"Qdrant delete failed: {e}")

    def _build_filter(self, filter_dict: Dict) -> Filter:
        """
        Build qdrant filter from dict with support for:
        - Exact match: {"source": "doc.pdf"}
        - Range queries: {"fiscal_year": {"gte": 2024, "lte": 2025}}
        - List values: {"source": ["doc1.pdf", "doc2.pdf"]}
        """
        from qdrant_client.models import Range, MatchAny
        
        conditions = []

        for key, value in filter_dict.items():
            if value is None:
                continue
            # range filter: {"fiscal_year": {"gte": 2024, "lte": 2025}}
            if isinstance(value, dict) and any(k in value for k in ['gte', 'lte', 'gt', 'lt']):
                range_params = {}
                if 'gte' in value:
                    range_params['gte'] = value['gte']
                if 'lte' in value:
                    range_params['lte'] = value['lte']
                if 'gt' in value:
                    range_params['gt'] = value['gt']
                if 'lt' in value:
                    range_params['lt'] = value['lt']
                    
                conditions.append(FieldCondition(
                    key=key,
                    range=Range(**range_params)
                ))
            # list values: {"source": ["doc1.pdf", "doc2.pdf"]}
            elif isinstance(value, list):
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchAny(any=value)
                ))
            # exact match (existing behavior)
            else:
                condition = FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                conditions.append(condition)

        return Filter(must=conditions) if conditions else None

    def _serialize_payload(self, payload: Dict) -> Dict:
        """Serialize complex metadata values to JSON strings"""
        import json

        serialized = {}

        for key, value in payload.items():
            # skip None values
            if value is None:
                continue

            # handle primitive types (str, int, float, bool)
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value

            # handle lists
            elif isinstance(value, list):
                # serialize to JSON if contains complex objects
                try:
                    json.dumps(value)  # test if serializable
                    serialized[key] = value
                except (TypeError, ValueError):
                    serialized[key] = json.dumps(value)
            # handle dicts
            elif isinstance(value, dict):
                # serialize to JSON string
                serialized[key] = json.dumps(value)
            # handle other types (serialize to string)
            else:
                serialized[key] = str(value)
        return serialized

    def _deserialize_payload(self, payload: Dict) -> Dict:
        """Deserialize JSON strings back to dicts"""
        import json

        deserialized = {}

        for key, value in payload.items():
            # try to deserialize JSON strings
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    deserialized[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    deserialized[key] = value
            else:
                deserialized[key] = value

        return deserialized


def create_qdrant_retriever(
    collection_name: str,
    embedding_function: Any,
    search_kwargs: Optional[Dict] = None,
) -> Any:
    """create a langchain-compatible retriever from qdrant"""
    from langchain_core.retrievers import BaseRetriever

    # get qdrant path from config
    qdrant_path = str(Config.Path.VECTOR_DB_DIR)
    # create vector store
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        embedding_function=embedding_function,
        path=qdrant_path,
        embedding_dim=768,  # gte-multilingual-base
    )
    # create retriever wrapper
    class QdrantRetriever(BaseRetriever):
        """langchain-compatible qdrant retriever"""
        vector_store: Any
        search_kwargs: Dict
        model_config = {"arbitrary_types_allowed": True}

        def _get_relevant_documents(self, query: str) -> List[Document]:
            """retrieve documents"""
            k = self.search_kwargs.get('k', 4)
            filter_dict = self.search_kwargs.get('filter', None)
            return self.vector_store.similarity_search(query, k=k, filter=filter_dict)

        async def _aget_relevant_documents(self, query: str) -> List[Document]:
            """async retrieve (fallback to sync)"""
            return self._get_relevant_documents(query)

    search_kwargs = search_kwargs or {'k': 4}

    return QdrantRetriever(
        vector_store=vector_store,
        search_kwargs=search_kwargs,
    )