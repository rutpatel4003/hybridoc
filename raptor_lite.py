from typing import List, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from sklearn.cluster import KMeans
from config import Config
import hashlib


@dataclass
class ClusterSummary:
    """Summary of a cluster of related chunks"""
    cluster_id: int
    summary: str
    num_chunks: int
    chunk_ids: List[str]
    source_files: List[str]


class RaptorLite:
    """
    Single-level hierarchical summarization for RAG.

    Process:
    1. Embed all chunks
    2. Cluster chunks using k-means
    3. Generate summary for each cluster
    4. Return summaries as new Document objects

    At retrieval time:
    - Vector search retrieves from both original chunks AND summaries
    - Summaries provide high-level context
    - Original chunks provide specific details
    """

    def __init__(
        self,
        embedder: Embeddings,
        llm = None,
        num_clusters: Optional[int] = None,
        min_cluster_size: int = 3,
        max_cluster_size: int = 50,
    ):
        self.embedder = embedder
        self.llm = llm
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

        # summary prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are summarizing a group of related text chunks from technical documents.

Create a comprehensive summary that:
1. Captures the main themes and key information
2. Preserves important technical details (numbers, formulas, names)
3. Maintains context about what topic/section this relates to
4. Is dense and information-rich (not generic)

The summary will be used for retrieval, so include relevant keywords and terminology.

Write in a natural paragraph style (3-5 sentences). Do NOT use bullet points or lists."""),
            ("human", """Summarize these related chunks:

{chunks}

Summary:"""),
        ])

    def _calculate_optimal_clusters(self, num_chunks: int) -> int:
        """
        Auto-calculate number of clusters based on corpus size.
        Heuristic: sqrt(n/2) with bounds
        """
        if num_chunks < self.min_cluster_size:
            return 1

        # square root heuristic (common for k-means)
        k = int(np.sqrt(num_chunks / 2))
        # apply bounds
        min_clusters = max(1, num_chunks // self.max_cluster_size)
        max_clusters = num_chunks // self.min_cluster_size
        return max(min_clusters, min(k, max_clusters))

    def _get_chunk_id(self, doc: Document) -> str:
        """Generate stable ID for a chunk"""
        md = doc.metadata or {}
        source = md.get('source', '')
        page = md.get('page', 0)
        content_hash = hashlib.md5(doc.page_content[:200].encode()).hexdigest()[:8]
        return f"{source}:p{page}:{content_hash}"

    def _cluster_chunks(self, chunks: List[Document], embeddings: np.ndarray) -> List[List[int]]:
        """
        Cluster chunks using k-means on embeddings.
        """
        num_chunks = len(chunks)
        k = self.num_clusters or self._calculate_optimal_clusters(num_chunks)

        if k >= num_chunks:
            # too few chunks - each chunk is its own cluster
            return [[i] for i in range(num_chunks)]

        # k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        # group chunk indices by cluster
        clusters: dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # filter out tiny clusters
        valid_clusters = [
            indices for indices in clusters.values()
            if len(indices) >= self.min_cluster_size
        ]

        print(f"Created {len(valid_clusters)} clusters (from {k} initial, filtered by min_size={self.min_cluster_size})")

        return valid_clusters

    def _generate_summary(self, cluster_chunks: List[Document]) -> str:
        """Generate summary for a cluster of chunks"""
        # concatenate chunk contents
        combined_text = "\n\n---\n\n".join([
            f"Chunk {i+1} (from {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}):\n{doc.page_content}"
            for i, doc in enumerate(cluster_chunks)
        ])

        # truncate if too long (LLM context limit)
        max_chars = 15000
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars] + f"\n\n[... truncated {len(combined_text) - max_chars} chars ...]"

        # generate summary
        if self.llm is None:
            from data_ingestor import get_chatbot_llm
            self.llm = get_chatbot_llm()

        chain = self.summary_prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"chunks": combined_text})

        # clean thinking tags
        if '</think>' in summary:
            summary = summary.split('</think>')[-1].strip()

        return summary

    def create_summaries(self, chunks: List[Document]) -> List[Document]:
        """
        Create cluster summaries and return as Documents.
        """
        if not chunks:
            return []

        if len(chunks) < self.min_cluster_size:
            print(f"  RAPTOR: Skipping - only {len(chunks)} chunks (below min_cluster_size={self.min_cluster_size})")
            return []

        print(f"\n{'='*60}")
        print(f"RAPTOR LITE: Creating hierarchical summaries")
        print(f"{'='*60}")
        print(f"Input chunks: {len(chunks)}")
        # 1. embed all chunks
        print("  Embedding chunks...")
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedder.embed_documents(texts)
        embeddings_array = np.array(embeddings)
        # 2. cluster chunks
        print("  Clustering...")
        clusters = self._cluster_chunks(chunks, embeddings_array)
        if not clusters:
            print("  RAPTOR: No valid clusters created")
            return []
        # 3. generate summaries
        print(f"  Generating {len(clusters)} summaries...")
        summary_docs = []
        for cluster_id, chunk_indices in enumerate(clusters):
            cluster_chunks = [chunks[i] for i in chunk_indices]
            # get metadata
            sources = list(set([doc.metadata.get('source', 'unknown') for doc in cluster_chunks]))
            pages = list(set([doc.metadata.get('page', 0) for doc in cluster_chunks]))
            chunk_ids = [self._get_chunk_id(doc) for doc in cluster_chunks]
            print(f"    Cluster {cluster_id+1}/{len(clusters)}: {len(cluster_chunks)} chunks from {sources[0]}")
            # generate summary
            try:
                summary_text = self._generate_summary(cluster_chunks)
                # create summary document
                summary_doc = Document(
                    page_content=summary_text,
                    metadata={
                        'source': sources[0] if len(sources) == 1 else f"{sources[0]} (+{len(sources)-1} more)",
                        'page': pages[0] if len(pages) == 1 else min(pages),  # Use first page
                        'content_type': 'raptor_summary',
                        'raptor_cluster_id': cluster_id,
                        'raptor_num_chunks': len(cluster_chunks),
                        'raptor_chunk_ids': chunk_ids,
                        'raptor_sources': sources,
                        'raptor_pages': pages,
                    }
                )
                summary_docs.append(summary_doc)
            except Exception as e:
                print(f"WARNING: Failed to generate summary for cluster {cluster_id}: {e}")
                continue

        print(f"Created {len(summary_docs)} summaries")
        print(f"{'='*60}\n")

        return summary_docs


def apply_raptor_lite(
    chunks: List[Document],
    embedder: Embeddings,
    llm: Optional[ChatLlamaCpp] = None
) -> List[Document]:
    """
    Convenience function to apply RAPTOR Lite and return enhanced chunk list.
    """
    if not Config.Preprocessing.ENABLE_RAPTOR_LITE:
        return chunks
    raptor = RaptorLite(
        embedder=embedder,
        llm=llm,
        num_clusters=getattr(Config.Raptor, 'NUM_CLUSTERS', None),
        min_cluster_size=getattr(Config.Raptor, 'MIN_CLUSTER_SIZE', 3),
        max_cluster_size=getattr(Config.Raptor, 'MAX_CLUSTER_SIZE', 50),
    )
    summaries = raptor.create_summaries(chunks)
    # return original chunks + summaries
    return chunks + summaries
