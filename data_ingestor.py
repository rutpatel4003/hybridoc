from typing import List
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
from langchain_chroma import Chroma
import hashlib

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
    return ChatOllama(model=Config.Preprocessing.LLM, temperature=0, keep_alive=-1)

def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model = Config.Preprocessing.RERANKER, top_n = Config.Chatbot.N_CONTEXT_RESULTS)

def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Preprocessing.EMBEDDING_MODEL)

def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
    messages = CONTEXT_PROMPT.format_messages(document=document, chunk=chunk)
    response = llm.invoke(messages)
    return response.content

def _create_chunks(document: Document) -> List[Document]:
    chunks = text_splitter.split_documents([document])
    if not Config.Preprocessing.CONTEXTUALIZE_CHUNKS:
        return chunks
    llm = create_llm()
    contextual_chunks = []
    for c in chunks:
        context = _generate_context(llm, document.page_content, c.page_content)
        chunk_with_context = f"{context}\n\n{c.page_content}"
        contextual_chunks.append(Document(page_content=chunk_with_context, metadata=c.metadata))
    return contextual_chunks

def _calculate_file_hash(content: str) -> str:
    """
    Calculate hash of file content for deduplication
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def ingest_files(files: List[File]) -> BaseRetriever:
    """
    Ingests into a Persistent Vector Database (Chroma)
    Implements 'Incremental Indexing' to skip files that are already indexed
    """
    #initialize embeddings
    embedding_model = create_embeddings()

    # connect to persistent db on disk
    vector_store = Chroma(
        collection_name='private-rag',
        embedding_function=embedding_model,
        persist_directory=str(Config.Path.VECTOR_DB_DIR)
    )

    # check for duplicated files
    try:
        existing_data = vector_store.get()
        existing_sources = {}
        if existing_data and 'metadatas' in existing_data:
            for m in existing_data['metadatas']:
                if m and 'source' in m:
                    source_name = m['source']
                    file_hash = m.get('content_hash', None)
                    existing_sources[source_name] = file_hash
        print(f'Found {len(existing_sources)} files in database')
        print(f'Files: {list(existing_sources.keys())}')
    except Exception as e:
        print(f'Error reading database: {e}')
        existing_sources = {}

    # filter new files only
    new_chunks = []
    skipped_files = []

    for f in files:
        file_hash = _calculate_file_hash(f.content)
        if f.name in existing_sources:
            stored_hash = existing_sources[f.name]
            if stored_hash == file_hash:
                print(f'Skipping {f.name} (already indexed)')
                continue
            else:
                print(f'File {f.name} content changed - reprocessing')

        # If it is a new file to process
        print(f"Indexing: {f.name}")
        doc = Document(f.content, metadata={'source': f.name, 'content_hash': file_hash})
        file_chunks = _create_chunks(doc)
        for chunk in file_chunks:
            chunk.metadata['content_hash'] = file_hash
        
        new_chunks.extend(file_chunks)

    if skipped_files:
        print(f'Loaded {len(skipped_files)} from cache.')

    # add new chunks to the db only
    if new_chunks:
        print(f'Adding {len(new_chunks)} new chunks to the Vector Database')
        vector_store.add_documents(new_chunks)

    # create vector retriever
    semantic_retriever = vector_store.as_retriever(
        search_kwargs={'k':Config.Preprocessing.N_SEMANTIC_RESULTS}
    )

    # create bm25 retriever 
    all_docs = []

    db_state = vector_store.get()
    stored_texts = db_state.get('documents', [])
    stored_metadatas = db_state.get('metadatas', [])
    if not stored_texts:
        raise ValueError('Database is empty! Please upload a document.')
    
    # reconstruct document objects for langchain
    global_corpus = []
    for t, m in zip(stored_texts, stored_metadatas):
        safe_m = m if m else {} # in case if metadata is none
        global_corpus.append(Document(page_content=t, metadata=safe_m))

    print(f'Building BM25 Index on {len(global_corpus)} total chunks')
    bm25_retriever = BM25Retriever.from_documents(global_corpus)
    bm25_retriever.k = Config.Preprocessing.N_BM25_RESULTS

    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return ContextualCompressionRetriever(base_compressor=create_reranker(), base_retriever=ensemble_retriever)



