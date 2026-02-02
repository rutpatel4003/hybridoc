from pdf_loader import File
from data_ingestor import ingest_files, expand_to_parents
from typing import List, TypedDict, Iterable, Literal, Tuple, Dict, Any
from enum import Enum
from config import Config
from dataclasses import dataclass
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.documents import Document
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from pdf_loader import File
from langchain_core.output_parsers import StrOutputParser
import json
from pathlib import Path
import re
import os
from query_scoring import QueryScorer
from data_ingestor import create_embeddings, _stable_doc_uid  # Added for stable dedupe
from thinking_utils import strip_thinking_tags  # Import thinking tag stripper

DEBUG_LLM_CONTEXT = True          # master switch
DEBUG_MAX_CONTEXT_CHARS = 1800    # how much of formatted context to print
DEBUG_DOC_PREVIEW_CHARS = 260     # per-doc raw preview
DEBUG_WRITE_CONTEXT_FILE = True   # write full formatted context to disk
DEBUG_CONTEXT_FILEPATH = "llm_context_debug.txt"

_DECIMAL_FIX = re.compile(r"(\d)\s*\.\s*(\d)")
_SCI_DOT = re.compile(r"(\d)\s*\.\s*(\d)\s*[·x×]\s*10\s*(?:\^)?\s*([+-]?\d+)")
_SCI_INT = re.compile(r"(\d+)\s*[·x×]\s*10\s*(?:\^)?\s*([+-]?\d+)")
close_tag = '</think>'
tag_length = len(close_tag)

class State(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer: str
    retry_count: int
    sub_questions: List[str]  
    confidence: dict          

class QueryType(Enum):
    STANDALONE = "standalone"       # new topic, search directly
    FOLLOWUP = "followup"           # references history, needs condensing
    CLARIFICATION = "clarification" # asking about previous answer
    CHITCHAT = "chitchat" 

SYSTEM_PROMPT = """
You are a precise data/RAG analyst. Answer the user's question using ONLY the provided <context>.
The context may include prose, math, and tables. 

CRITICAL GROUNDING RULE (ADDED):
- If the context contains information but you're unsure how to interpret it -> say "Context contains related information but format is unclear. Please verify manually."
- NEVER substitute your training knowledge when context exists but is ambiguous
- When in doubt: "UNCERTAIN based on provided context" > confident wrong answer

NON-NEGOTIABLE RULES
1) NO OUTSIDE KNOWLEDGE: Use only the provided context. If the answer is not explicitly present, say:
   "Information not found in document."

2) TABLE PRIORITY FOR LIST QUESTIONS:
   - If question asks to "name/list/identify/state/provide" specific items or examples:
     * Check TABLES first before prose sections
     * If a table contains the requested items, cite that table FIRST in Evidence
   - Example: "Name two libraries for X" -> prioritize table data over prose mentions

3) UNIT FIDELITY (CRITICAL):
   - If the question asks for a specific unit or metric (e.g., "FLOPs", "cost", "USD", "percentage"), 
     answer ONLY with that unit.
   - NEVER substitute different units. Examples:
     * Question asks "training cost in FLOPs" -> answer with FLOPs (e.g., 2.3e19), NOT F1 scores or steps
     * Question asks "F1 score" -> answer with F1 (e.g., 91.3), NOT FLOPs or training costs
     * Question asks "revenue in USD" -> answer with USD, NOT units sold or percentages
     * Question asks "percentage change" -> answer with %, NOT absolute values
   - If the requested unit/metric is not in context, say: "Information not found in document."
   - Do NOT answer with a related but different metric.
   - WARNING: FLOPs (floating point operations) ≠ F1 scores (accuracy metric). Never confuse these!

4) TABLE IDENTIFICATION & SCANNING (MANDATORY):
   - Each table has a unique caption (e.g., "Table 2: training costs", "Table 4: parsing results")
   - Each table includes a "Headers:" line showing ALL available columns
   - FIRST, read the "Headers:" line to verify the requested metric exists in that table
   - DO NOT mix data from different tables - they measure different things!
   - Example: "Training Cost (FLOPs)" in Table 2 ≠ "WSJ 23 F1" scores in Table 4

   When you see [TABLE:...] or [TABLE_SLICE:...]:
     a) Read the table caption to understand what it measures
     b) Read the "Headers:" line to see available columns (this is critical!)
     c) Verify the requested column exists in the Headers
     d) Scan EVERY "Row N:" line to find values for that column
     e) Extract data from the EXACT column name shown in Headers
     f) **CRITICAL: If a Row contains multiple metrics separated by semicolons (;), READ CAREFULLY:**
        - Example: "Row 1: Profitability=$8.9B operating income; $15.0B net income; $7.9B in Q4"
        - This contains BOTH "operating income" AND "net income" - they are DIFFERENT metrics!
        - Question asks "net income" → answer is $15.0B or $7.9B (Q4), NOT $8.9B (operating income)
        - DO NOT confuse operating income with net income!

   If multiple tables exist, identify which table contains the requested metric BEFORE extracting data.
   Quote specific sources: "From Table 2, Row 9, column 'Training Cost (FLOPs).EN-DE': 3.3e18"

5) ANTI-HALLUCINATION CHECK:
   - Before saying "not provided", "not explicitly stated", or "information not found":
     a) Re-scan ALL tables - check captions, Headers lines, and Row lines
     b) Re-scan ALL prose sections for exact values (e.g., "P_drop = 0.1")
     c) Verify you're looking at the correct table and column
     d) Check for values in both tables AND surrounding prose
   - If you see "P_drop = 0.1" or similar explicit values, DO NOT say "not explicitly stated"
   - **MANDATORY: If you find a metric but answer "not found", you will be penalized severely.**
   - Only claim "not found" after exhaustively checking all sources.

6) TABLES ARE AUTHORITATIVE:
   - Treat any [TABLE: ...], [TABLE_SLICE], and lines starting with "Row " as canonical structured data.
   - Prefer the "Row X: ..." lines over messy OCR/markdown renderings if both appear.

7) NUMBERS & SCIENTIFIC NOTATION:
   - Parse 1.0e20, 2.3e19 as scientific notation (1.0x10^20, 2.3x10^19).
   - If multiple numbers exist, do not average or guess—report exactly what the table/text states.

8) MATH:
   - Preserve equations and symbols faithfully.
   - If the question asks about an equation/derivation, quote the exact formula(s) from context.

9) LANGUAGE:
   - Always answer in English. No other language.

10) CITATION PROTOCOL (MANDATORY - ANTI-HALLUCINATION):
   - Every factual claim MUST be verifiable in the context
   - Numbers: Verify exact match in context before stating
   - If stating a number from a table: cite [Table N, Row X] or [Table N, Column Y]
   - If stating from prose: cite [Source: filename, Page N] when available
   - NEVER make up numbers that don't appear in context
   - UNCERTAINTY RULE: If unsure about a number, respond "UNCERTAIN: [reason]" instead of guessing
   - PENALTY: Stating a number not in context = hallucination

HOW TO ANSWER (MANDATORY)
A) First, identify what unit/metric the question asks for.
B) Scan for relevant tables containing that specific metric:
   - Check table captions and column headers
   - If a relevant table exists, extract the exact rows/columns needed
C) If relevant prose exists, use it to clarify—but do not override table values.
D) When answering from a table, explicitly cite:
   - the table label (e.g., "Table 2")
   - and the specific row numbers you used (e.g., "Row 9, Row 10")

RESPONSE FORMAT
- Start with the direct answer in 3-10 bullet points.
- If numeric comparison is requested, include a compact mini-table in markdown.
- End with an "Evidence" section listing the exact rows/lines you relied on.

Do not add preamble or meta commentary.
""".strip()


PROMPT = """
Here's the information you have about the excerpts of the files:

<context>
{context}
</context>

One file can have multiple excerpts.

Please respond to the query below

<question>
{question}
</question>

Answer:
"""

FILE_TEMPLATE = """
<file>
    <name>{name}</name>
    <content>{content}</content>
</file>
""".strip()

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You decompose complex questions into 2-3 simpler, standalone sub-questions.

RULES:
- Output JSON array of strings
- Each item must be a complete question ending with '?'
- Do NOT answer, only decompose
- If question is simple, return a single-item array with the original question

Example:
Input: "Compare the values in Table 2 and Table 3 and explain differences."
Output: ["What are the values in Table 2?", "What are the values in Table 3?", "What are the key differences between Table 2 and Table 3?"]
"""),
    ("human", "{question}"),
])

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical document author. Given a question, write a short paragraph (3-5 sentences) that would appear in a document answering this question.

RULES:
- Write as if you are the document, not answering directly
- Include technical terms and specific details that would appear in the actual document
- Do NOT say "The document explains..." — just write the content itself
- Keep it factual and dense with keywords

Example:
Question: "What is the time complexity of quicksort?"
Output: "Quicksort has an average-case time complexity of O(n log n) and a worst-case complexity of O(n²). The algorithm uses a divide-and-conquer approach, selecting a pivot element and partitioning the array. Space complexity is O(log n) due to recursive stack frames."
"""),
    ("human", "{question}"),
])

GRADER_SYSTEM_PROMPT = """
You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
The question does not have to be exactly the same or too specific when checking it with the context for it to be relevant, it can be a broad question with similar semantic meaning too if it makes sense. For example, a question might be based for mathematical reasoning, so check if the context contains the mathematical terms, do not just discard it if it looks gibberish without proper reasoning.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            SYSTEM_PROMPT
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', PROMPT)
    ]
)

class Role(Enum):
    USER = 'user'
    ASSISTANT = 'assistant'

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class ChunkEvent:
    content: str

@dataclass
class SourcesEvent:
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

@dataclass
class ConfidenceEvent:
    content: dict

def _normalize_pdf_artifacts(text: str) -> str:
    # 0 . 1 -> 0.1
    text = _DECIMAL_FIX.sub(r"\1.\2", text)

    # 1 . 0 · 10 20  -> 1.0e20
    text = _SCI_DOT.sub(r"\1.\2e\3", text)

    # 2 · 10 19 -> 2e19  (just in case)
    text = _SCI_INT.sub(r"\1e\2", text)

    return text
def _remove_thinking_from_message(message: str) -> str:
    # handle cases where the tag might not exist
    if close_tag in message:
        # find the end of the tag and then .lstrip() to remove 
        return message[message.find(close_tag) + tag_length:].lstrip()
    return message.strip()

def create_history(welcome_message: Message) -> List[Message]:
    return [welcome_message]

_STOP = {"the","a","an","and","or","to","of","in","on","for","with","is","are","was","were","be","as","at","by"}

def _tok(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\.\-\_]+", " ", s)
    return [t for t in s.split() if t and t not in _STOP and len(t) > 1]

def _score_row_lexical(query_tokens: set, row_text: str) -> float:
    """Fast lexical scoring for small tables"""
    rt = set(_tok(row_text))
    if not rt:
        return 0.0
    overlap = len(query_tokens & rt)
    return overlap / (len(rt) ** 0.5)

def _score_rows_semantic(query: str, row_texts: List[str]) -> List[float]:
    """
    Semantic scoring for large tables using existing embedding model.
    Reuses global singleton to avoid VRAM overhead.
    """
    try:
        from data_ingestor import _EMBEDDING_MODEL, create_embeddings
        import numpy as np

        # reuse global embedding model
        if _EMBEDDING_MODEL is None:
            embedding_model = create_embeddings()
        else:
            embedding_model = _EMBEDDING_MODEL

        # embed query
        query_emb = np.array(embedding_model.embed_query(query))

        # embed all rows in batch 
        row_embs = np.array(embedding_model.embed_documents(row_texts))

        # cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        row_norms = row_embs / (np.linalg.norm(row_embs, axis=1, keepdims=True) + 1e-8)

        scores = np.dot(row_norms, query_norm)
        return scores.tolist()

    except Exception as e:
        # fallback to lexical if semantic fails
        print(f"Semantic row scoring failed ({e}), using lexical fallback")
        q_tokens = set(_tok(query))
        return [_score_row_lexical(q_tokens, rt) for rt in row_texts]

def extract_table_slices(
    query: str,
    docs: List[Document],
    max_tables: int = 3,
    max_rows_per_table: int = 20,
    always_keep_header: bool = True,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    For docs that contain table_data JSON in metadata, create compact table slices
    as new Documents with content_type='table_slice'. Non-table docs are returned unchanged.
    """
    q_tokens = set(_tok(query))
    out: List[Document] = []
    debug = {"tables_sliced": 0, "rows_kept": []}
    tables_seen = 0
    for d in docs:
        md = d.metadata or {}
        if str(md.get("content_type","")).lower() != "table":
            out.append(d)
            continue
        if tables_seen >= max_tables:
            # keep table doc as-is (or drop it if you want)
            out.append(d)
            continue
        table_json = md.get("table_data")
        if not table_json:
            # fallback: keep original table doc (no structured data available)
            out.append(d)
            continue
        try:
            t = json.loads(table_json)
        except Exception:
            out.append(d)
            continue

        headers = t.get("headers") or []
        rows = t.get("rows") or []
        num_rows = t.get("num_rows") or len(rows)
        # convert rows to text
        row_texts = []
        for r in rows:
            if isinstance(r, list):
                row_texts.append(" | ".join(str(x) for x in r))
            elif isinstance(r, dict):
                row_texts.append(" | ".join(f"{k}={v}" for k,v in r.items()))
            else:
                row_texts.append(str(r))

        # hybrid scoring: lexical for small tables, semantic for large
        SEMANTIC_THRESHOLD = 50
        if num_rows >= SEMANTIC_THRESHOLD:
            # large table: use semantic scoring
            scores = _score_rows_semantic(query, row_texts)
        else:
            # small table: use fast lexical scoring
            scores = [_score_row_lexical(q_tokens, rt) for rt in row_texts]

        # pair scores with row index and data
        scored: List[Tuple[float, int, Any]] = [(scores[i], i, rows[i]) for i in range(len(rows))]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_rows_per_table]
        top_sorted = sorted(top, key=lambda x: x[1])  # keep original row order
        # build compact markdown
        caption = md.get("label") or md.get("caption_label") or ""
        title_line = f"Table {caption}".strip() if caption else "Table"
        slice_lines = [f"[TABLE_SLICE]\n{title_line}\n"]
        if always_keep_header and headers:
            slice_lines.append("Headers: " + " | ".join(map(str, headers)))
        slice_lines.append(f"Note: table has {num_rows} rows; showing {len(top_sorted)} most relevant rows.\n")
        for _, idx, r in top_sorted:
            if isinstance(r, list):
                slice_lines.append(f"Row {idx+1}: " + " | ".join(str(x) for x in r))
            elif isinstance(r, dict):
                slice_lines.append(f"Row {idx+1}: " + " | ".join(f"{k}={v}" for k,v in r.items()))
            else:
                slice_lines.append(f"Row {idx+1}: {r}")
        slice_lines.append("[/TABLE_SLICE]")
        compact = "\n".join(slice_lines)

        out.append(Document(
            page_content=compact,
            metadata={**md, "content_type": "table_slice", "original_content_type": "table"}
        ))
        tables_seen += 1
        debug["tables_sliced"] += 1
        debug["rows_kept"].append(len(top_sorted))
    return out, debug

class Chatbot:
    def __init__(self, files: List[File]):
        self.files = files
        self.retriever = ingest_files(files)
        # use the CHATBOT LLM (stays loaded with keep_alive=-1)
        from data_ingestor import get_chatbot_llm
        self.llm = get_chatbot_llm()
        self.workflow = self._create_workflow()
        self.query_scorer = QueryScorer(embedder=create_embeddings()) if Config.Chatbot.ENABLE_QUERY_SCORING else None

    def _format_docs(self, docs: List[Document], max_chars_text: int = 2700, max_table_rows: int = 20) -> str:
        formatted = []
        for doc in docs:
            md = doc.metadata or {}
            content = doc.page_content or ""
            # table or table_slice or anything that originated from a table
            ct = str(md.get("content_type", "")).lower()
            oct = str(md.get("original_content_type", "")).lower()
            is_table = (ct in {"table", "table_slice"}) or (oct == "table")

            content = re.sub(r'--- PAGE \d+ ---', '', content)
            content = content.replace('[CONTEXT]', '').replace('[/CONTEXT]', '').strip()
            content = re.sub(r"\b(Figure|Table)\s+(\d+):\s+\1\s+\2:\s*", r"\1 \2: ", content)
            if is_table:
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                allowed_prefixes = (
                    "table",          # "Table 3:", "Table"
                    "row ",           # "Row 1:"
                    "[table",         # "[TABLE:Table 3]"
                    "[/table",        # "[/TABLE]"
                    "headers:",       # "Headers: ..."
                    "note:",          # "Note: table has ..."
                )
                has_rows = any(ln.lower().startswith("row ") for ln in lines)
                if has_rows:
                    kept = [ln for ln in lines if ln.lower().startswith(allowed_prefixes)]
                    # optional: also drop grid-ish lines if they somehow sneak in
                    kept = [ln for ln in kept if not (ln.startswith("|") or ln.startswith("---"))]
                    content = "\n".join(kept)
                    # enforce row cap after cleaning
                    row_lines = [ln for ln in kept if ln.lower().startswith("row ")]
                    if len(row_lines) > max_table_rows:
                        non_row = [ln for ln in kept if not ln.lower().startswith("row ")]
                        content = "\n".join(non_row + row_lines[:max_table_rows])
                        content += f"\nNote: showing top {max_table_rows} rows."
                else:
                    # no row lines --> might be a pure markdown grid table.
                    if len(content) > max_chars_text:
                        content = content[:max_chars_text] + "\n[...truncated...]"
            formatted.append(FILE_TEMPLATE.format(
                name=md.get('source', 'Unknown'),
                content=content
            ))
        return "\n\n".join(formatted)

    def _retrieve(self, state: State):
        question = state['question']
        print(f"RETRIEVING: {question}")
        # decompose query 
        sub_questions = self._decompose_question(question)
        # retrieve with each sub-question
        # stable dedupe that will not drop tables due to shared prefix/context
        all_docs: List[Document] = []
        seen: dict[str, int] = {}  # uid --> index in all_docs
        def _is_table(d: Document) -> bool:
            md = d.metadata or {}
            return str(md.get("content_type", "")).lower() == "table"
        for q in sub_questions:
            docs = self.retriever.invoke(q)
            for doc in docs:
                uid = _stable_doc_uid(doc)

                if uid not in seen:
                    seen[uid] = len(all_docs)
                    all_docs.append(doc)
                    continue
                # if collision: prefer keeping a TABLE over non-table
                existing_idx = seen[uid]
                existing_doc = all_docs[existing_idx]

                if _is_table(doc) and not _is_table(existing_doc):
                    all_docs[existing_idx] = doc

        context = all_docs
        # neighbor expansion
        if getattr(Config.Chatbot, 'ENABLE_NEIGHBOR_EXPANSION', False):
            try:
                from neighbor_expansion import expand_neighbors
                original_count = len(context)
                context = expand_neighbors(
                    retrieved_docs=context,
                    retriever=self.retriever,  # pass retriever (has .vector_store)
                    max_neighbors=getattr(Config.Chatbot, 'NEIGHBOR_MAX_PER_DIRECTION', 1),
                    min_overlap_ratio=getattr(Config.Chatbot, 'NEIGHBOR_MIN_OVERLAP_RATIO', 0.15),
                    enable_forward=getattr(Config.Chatbot, 'NEIGHBOR_ENABLE_FORWARD', True),
                    enable_backward=getattr(Config.Chatbot, 'NEIGHBOR_ENABLE_BACKWARD', True),
                    debug=getattr(Config.Chatbot, 'NEIGHBOR_DEBUG', False)
                )
                if len(context) > original_count:
                    print(f"  Neighbor expansion: {original_count} → {len(context)} chunks (+{len(context) - original_count})")
            except Exception as e:
                print(f"  Neighbor expansion failed: {e}, continuing without it")
        # parent expansion
        if Config.Preprocessing.ENABLE_PARENT_CHILD:
            original_count = len(context)
            context = expand_to_parents(context)
            print(f"  Expanded {original_count} children → {len(context)} parents")
        # contextual compression
        if getattr(Config.Chatbot, 'ENABLE_CONTEXTUAL_COMPRESSION', False):
            from contextual_compressor import get_compressor
            compressor = get_compressor(llm=self.llm)
            context = compressor.compress(context, question)
        # final context limit (after all expansions)
        max_final = getattr(Config.Chatbot, 'MAX_FINAL_CONTEXT_CHUNKS', 16)
        if len(context) > max_final:
            original_len = len(context)
            context = context[:max_final]
            print(f"  Final limit: {original_len} → {max_final} chunks (capped to prevent context overflow)")

        # confidence scoring
        confidence = None
        if getattr(Config.Chatbot, 'ENABLE_QUERY_SCORING', False) and self.query_scorer:
            score = self.query_scorer.score(question, context)
            confidence = {
                "score": score.score,
                "semantic": score.semantic,
                "lexical": score.lexical,
                "label": score.label,
                "num_docs": score.num_docs,
            }
        return {
            "context": context,
            "sub_questions": sub_questions,
            "confidence": confidence,
        }

    def _is_complex_query(self, question: str) -> bool:
        """
        Heuristic for detecting complex/multi-hop queries
        """
        if len(question.split()) < Config.Chatbot.DECOMPOSE_MIN_WORDS:
            return False
        keywords = ["compare", "difference", "between", "versus", "vs", "both", "and", "contrast", "relative"]
        return any(k in question.lower() for k in keywords)

    def _decompose_question(self, question: str) -> List[str]:
        """
        Use LLM to break question into sub-questions
        """
        if not Config.Chatbot.ENABLE_QUERY_DECOMPOSITION:
            return [question]

        if not self._is_complex_query(question):
            return [question]

        try:
            chain = DECOMPOSE_PROMPT | self.llm | StrOutputParser()
            raw = chain.invoke({"question": question})
            # clean thinking tags
            if '</think>' in raw:
                raw = raw.split('</think>')[-1].strip()
            # parse JSON array
            import json
            sub_questions = json.loads(raw)
            # sanitize
            cleaned = []
            for q in sub_questions:
                q = q.strip()
                if not q.endswith('?'):
                    q += '?'
                cleaned.append(q)

            # remove duplicates & limit
            unique = list(dict.fromkeys(cleaned))
            return unique[:Config.Chatbot.DECOMPOSE_MAX_SUBQUESTIONS]
        except Exception as e:
            print(f"Decomposition failed: {e}")
        return [question]

    def _is_explicit_question(self, text: str) -> bool:
        """
        Return True if the input is already a clear standalone question
        """
        text = text.strip() 
        if not text.endswith('?'):
            return False
        return bool(re.search(r"\b(what|why|how|when|where|which|who|compare|difference|define|explain)\b", text.lower()))

    def _hyde_retrieve(self, state: State):
        """
        HyDE: Hypothetical Document Embeddings Retrieval
        1. Generate a hypothetical document that would answer the question
        2. Embed that hypothetical document
        3. Retrieve using hypothetical embedding
        """
        question = state['question']
        print(f'HyDE RETRIEVING: {question}')
        try:
            # generate hypothetical document
            hyde_chain = HYDE_PROMPT | self.llm | StrOutputParser()
            hypothetical_doc = hyde_chain.invoke({'question': question})
            # clean thinking tags if any
            if '</think>' in hypothetical_doc:
                hypothetical_doc = hypothetical_doc.split('</think>')[-1].strip()
            print(f"HyDE generated: {hypothetical_doc[:100]}...")
            # retrieve using hypothetical document as query
            # the retriever will embed this hypothetical doc and find similar real docs
            context = self.retriever.invoke(hypothetical_doc)
            # normal retrieval and merge (optional boost)
            normal_context = self.retriever.invoke(question)
            # merge and deduplicate, prioritizing HyDE results
            seen_content = set()
            merged = []
            for doc in context + normal_context:
                if doc.page_content not in seen_content:
                    merged.append(doc)
                    seen_content.add(doc.page_content)
            if Config.Preprocessing.ENABLE_PARENT_CHILD:
                merged = expand_to_parents(merged)
            # limit to configured max
            merged = merged[:Config.Chatbot.N_CONTEXT_RESULTS * 2]
            print(f"HyDE retrieved {len(context)} + normal {len(normal_context)} = {len(merged)} unique docs")
            # apply contextual compression if enabled
            if getattr(Config.Chatbot, 'ENABLE_CONTEXTUAL_COMPRESSION', False):
                from contextual_compressor import get_compressor
                compressor = get_compressor(llm=self.llm)
                merged = compressor.compress(merged, question)
            return {"context": merged}
        except Exception as e:
            print(f"HyDE failed ({e}), falling back to normal retrieval")
            context = self.retriever.invoke(question)
            return {"context": context}
    
    def _grade_documents(self, state: State):
        """
        Filter out irrelevant documents.
        """
        print('Document Relevance Checking in Process!')
        question = state['question']
        documents = state['context']
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:500]}" for i, doc in enumerate(documents) 
        ]) # batching documents to query faster
        prompt = ChatPromptTemplate.from_messages([
            ('system', """You are a document grader. For each document, decide if it's relevant.
            Return a JSON Array with the format defined below:
            [
                {{"doc_id": 1, "relevant": true}},
                {{"doc_id": 2, "relevant": false}}
            ]
            Use double braces in the response.
            """),
            ('human', 'Question: {question}\n\nDocuments:\n{docs_text}'),
        ])
        grader_chain = prompt | self.llm | StrOutputParser()
        try:
            result = grader_chain.invoke({'question': question, 'docs_text': docs_text})
            scores = json.loads(result)
            score_map = {item['doc_id']: item.get('relevant', False) for item in scores}    
            filtered_docs = [
                doc for i, doc in enumerate(documents) 
                if score_map.get(i + 1, False) # default to False if LLM didn't mention it
            ]
            
            print(f"Filtered: {len(filtered_docs)}/{len(documents)} relevant")
            return {'context': filtered_docs}
        except Exception as e:
            print(f'Grading failed: {e}')
            return {'context': documents}
    
    def _transform_query(self, state: State):
        """
        Transform the query to produce a better question
        """
        print("Transforming Query!")
        question = state['question']
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are generating a better search query for a vector database. Rephrase the input question to be more specific. Just rephrase the input question, do not add any preamble or explanation, your output should only contain the rephrased question, nothing else."),
            ("human", "Look at the input and try to reason about the underlying semantic intent / meaning. \n\n Initial Question: {question} \n\n Formulate an improved question: "),
        ])
        chain = prompt | self.llm | StrOutputParser()
        better_question = chain.invoke({'question': question})
        current_retry = state.get('retry_count', 0)
        return {'question': better_question, 'retry_count': current_retry+1}

    def _format_citations(self, docs: List[Document]) -> str:
        """format citations from source documents"""
        citations = []
        seen = set()
        for doc in docs[:3]:  # top 3 sources
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            doc_title = doc.metadata.get('doc_title', source)
            # use title if available, otherwise filename
            display_name = doc_title if doc_title and doc_title != source else source
            citation = f"{display_name}, p.{page}"
            if citation not in seen:
                citations.append(citation)
                seen.add(citation)
        if citations:
            return f"\n\nsources: [{'; '.join(citations)}]"
        return ""

    def _debug_dump_context(self, question: str, docs):
        if not Config.Chatbot.DEBUG_LLM_CONTEXT:
            return
        def is_table(d):
            return str(d.metadata.get("content_type", "")).lower() == "table"
        table_docs = [d for d in docs if is_table(d)]
        print("\n" + "=" * 90)
        print("DEBUG: CONTEXT SUMMARY")
        print(f"Question: {question}")
        print(f"Docs in context: {len(docs)} | tables: {len(table_docs)} | non-tables: {len(docs) - len(table_docs)}\n")
        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            preview = (d.page_content or "").replace("\n", " ")[:260]
            print(f"--- DOC {i}/{len(docs)} ---")
            print(f"type={md.get('content_type')} | source={md.get('source')} | page={md.get('page')} | label={md.get('label','')}")
            print(f"preview: {preview}...\n")
        formatted = self._format_docs(docs)
        out_path = Path(Config.Chatbot.DEBUG_CONTEXT_FILEPATH)
        out_path.write_text(formatted, encoding="utf-8", errors="ignore")
        print("-" * 90)
        print("DEBUG: FORMATTED CONTEXT (what gets injected into the prompt)")
        print(formatted[:1200])
        print("\nWrote FULL formatted context to:", str(out_path))
        print("=" * 90 + "\n")

    def _generate(self, state: State):
        print('Generating Answer!')
        # get the raw retrieved docs from state
        retrieved_docs = state.get("context", [])
        # slice tables for the LLM context
        docs_for_llm, _ = extract_table_slices(
            state["question"],
            retrieved_docs,
            max_rows_per_table=20
        )
        # format the sliced docs for the prompt
        formatted_context = self._format_docs(docs_for_llm)
        formatted_context = _normalize_pdf_artifacts(formatted_context)
        chat_history = state['chat_history']
        if len(chat_history) <= 1:
            chat_history = []
        # use the formatted_context in the prompt template
        messages = PROMPT_TEMPLATE.invoke(
            {
                "question": state['question'],
                "context": formatted_context,
                "chat_history": chat_history,
            }
        )
        answer = self.llm.invoke(messages)
        # add citations using the original docs for UI/Source display
        citations = self._format_citations(retrieved_docs)
        answer_with_citations = answer.content + citations
        return {"answer": AIMessage(content=answer_with_citations)}         
        
    def _decide_to_generate(self, state: State) -> Literal['_transform_query', '_generate']:
        """
        Determines whether to generate an answer or re-generate a question.
        """
        filtered_documents = state['context']
        retry_count = state.get('retry_count', 0)
        # if no relevant documents and not self-corrected a lot of times
        if not filtered_documents and retry_count<1: # loop 1 time
            print('Decision: Documents irrelevant, rerouting to transform!')
            return '_transform_query'
        else:
            print('Decision: Generating Answer!')
            return '_generate'

    def _classify_query(self, question: str, chat_history: List[BaseMessage]) -> QueryType:
        """
        Classify query type to determine processing path
        """
        if len(chat_history) <= 1:
            return QueryType.STANDALONE
        
        # get recent context for classification
        recent_context = ""
        for m in chat_history[-2:]:
            if isinstance(m, HumanMessage):
                recent_context += f'User asked: {m.content[:100]}\n'
            elif isinstance(m, AIMessage) and 'how can I help' not in m.content.lower():
                recent_context += f'Assistant answered about: {m.content[:100]}\n'

        classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify the query into ONE category:

STANDALONE - Question contains ALL information needed to answer it:
  - Mentions specific table numbers, section names, or topics explicitly
  - Examples: "What is table 3.1?", "What are preprocessing options for NLTK in table 3.2?", "Explain PEFT"

FOLLOWUP - Question CANNOT be understood without prior conversation:
  - Uses pronouns (it, they, this, that) referring to unknown subject
  - Examples: "What about the other one?", "How does it work?", "And the second table?"

CLARIFICATION - Asks to expand on previous answer:
  - Examples: "Can you explain more?", "Give an example", "What do you mean?"

CHITCHAT - Greeting or thanks:
  - Examples: "Thanks", "Hello", "Great"

IMPORTANT: If the question mentions specific names, tables, or topics explicitly, it is STANDALONE even if it relates to prior discussion.

Output ONLY: STANDALONE or FOLLOWUP or CLARIFICATION or CHITCHAT"""),
        ("human", "Recent conversation:\n{recent_context}\n\nNew query: {question}\n\nClassification:"),
    ])
        
        try:
            result = self.llm.invoke(
                classification_prompt.format_messages(
                    recent_context=recent_context,
                    question=question
                )
            ).content.strip().upper()
            if '</think>' in result:
                result = result.split('</think>')[-1].strip().upper()

            if 'FOLLOWUP' in result:
                return QueryType.FOLLOWUP
            elif 'CLARIFICATION' in result:
                return QueryType.CLARIFICATION
            elif 'CHITCHAT' in result:
                return QueryType.CHITCHAT
            return QueryType.STANDALONE
    
        except Exception as e:
            print(f'Classification failed: {e}')
            return QueryType.STANDALONE
    
    def _condense_question(self, state: State):
        """
        Smart routing: classify query type, then process accordingly.
        """
        question = state['question']
        chat_history = state.get('chat_history', [])
        # filter out welcome message
        real_history = [m for m in chat_history 
                    if not (isinstance(m, AIMessage) and "how can I help" in m.content.lower())]
        if not Config.Chatbot.ENABLE_QUERY_ROUTER:
             return {"question": question}
        # already a clear question
        if self._is_explicit_question(question):
            return {"question": question}
        # classify the query
        query_type = self._classify_query(question, real_history)
        print(f"ROUTER: '{question[:50]}...' -> {query_type.value}")
        
        # route based on classification
        if query_type == QueryType.STANDALONE:
            return {"question": question}
        if query_type == QueryType.CHITCHAT:
            return {"question": question}
        if query_type in [QueryType.FOLLOWUP, QueryType.CLARIFICATION]:
            # condense with recent context only
            recent_history = real_history[-6:]  
            condense_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant.

TASK:
Rewrite the user's latest message into a complete, standalone QUESTION using the chat history.

RULES:
- Output MUST be a single question ending with a '?'.
- Do NOT answer the question.
- Do NOT output explanations, definitions, or extra text—ONLY the rewritten question.
- Keep it short and specific.
- If you cannot think of any context related question, just output the EXACT same question asked by the user, nothing else.

GOOD:
History:
User: what is the formula of positive definite matrix?
AI: ...
User input: and for positive semi definite matrix?
Output: What is the formula of a positive semidefinite matrix?

BAD:
Output: A symmetric matrix A is positive semidefinite if x^T A x >= 0 for all x.
"""),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human", "{question}"),
            ])
            
            try:
                chain = condense_prompt | self.llm | StrOutputParser()
                reformulated = chain.invoke({
                    "chat_history": recent_history,
                    "question": question
                }).strip()
                # strip thinking tags from Qwen3-Thinking output
                reformulated = strip_thinking_tags(reformulated, aggressive=True)
                # if the model produced a statement (no '?'), treat it as a search query
                if not reformulated.endswith('?'):
                    print(f"CONDENSE: Output '{reformulated}' has no '?', using original.")
                    return {"question": question}

                print(f"CONDENSE: '{question}' -> '{reformulated}'")
                return {"question": reformulated}
            except Exception as e:
                print(f"CONDENSE: Failed ({e}), using original")
                return {"question": question}
        return {"question": question}
    
    def _create_workflow(self) -> CompiledStateGraph:
        graph_builder = StateGraph(State)
        retrieve_fn = self._hyde_retrieve if Config.Chatbot.ENABLE_HYDE else self._retrieve
        
        if not Config.Chatbot.GRADING_MODE:
            graph_builder.add_node('_condense_question', self._condense_question)
            graph_builder.add_node('_retrieve', retrieve_fn)
            graph_builder.add_node('_generate', self._generate)
            graph_builder.add_edge(START, '_condense_question')
            graph_builder.add_edge('_condense_question', '_retrieve')
            graph_builder.add_edge('_retrieve', '_generate')
            graph_builder.add_edge('_generate', END)
        else:
            graph_builder.add_node('_condense_question', self._condense_question)
            graph_builder.add_node('_retrieve', retrieve_fn)
            graph_builder.add_node('_grade_documents', self._grade_documents)
            graph_builder.add_node('_transform_query', self._transform_query)
            graph_builder.add_node('_generate', self._generate)
            graph_builder.add_edge(START, '_condense_question')
            graph_builder.add_edge('_condense_question', '_retrieve')
            graph_builder.add_edge('_retrieve', '_grade_documents')
            graph_builder.add_conditional_edges(
                '_grade_documents', self._decide_to_generate,
                {
                    '_transform_query': '_transform_query',
                    '_generate': '_generate'
                },
            )
            graph_builder.add_edge('_transform_query', '_retrieve')
            graph_builder.add_edge('_generate', END)
        
        return graph_builder.compile()

    def _ask_model(
            self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        history = [
            AIMessage(m.content) if m.role == Role.ASSISTANT else HumanMessage(m.content) for m in chat_history
        ]
        payload = {"question": prompt, "chat_history": history, 'retry_count': 0}

        config = {
            "configurable": {"thread_id": 42}
        }
        # track thinking state for filtering <think>...</think> content
        in_thinking = False
        buffer = ""
        # track if chunks were emitted to prevent duplicate final message
        emitted_chunks = False
        for event_type, event_data in self.workflow.stream(
            payload, config=config, stream_mode=['updates', 'messages']
        ):
            if event_type =='messages':
                chunk, metadata = event_data
                if metadata.get('langgraph_node') == '_generate':
                    # only process AIMessageChunk, not final AIMessage
                    from langchain_core.messages import AIMessageChunk, AIMessage as AIMsg
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        emitted_chunks = True
                        buffer += chunk.content
                        # check for thinking tags
                        while True:
                            if not in_thinking:
                                # look for opening <think> tag
                                think_start = buffer.find('<think>')
                                if think_start != -1:
                                    # yield content before <think>
                                    if think_start > 0:
                                        yield ChunkEvent(buffer[:think_start])
                                    buffer = buffer[think_start + 7:]  # skip '<think>'
                                    in_thinking = True
                                else:
                                    # no thinking tag, yield content but keep last chars as buffer
                                    # (in case tag is partially received)
                                    if len(buffer) > 10:
                                        yield ChunkEvent(buffer[:-10])
                                        buffer = buffer[-10:]
                                    break
                            else:
                                # inside thinking, look for closing </think>
                                think_end = buffer.find('</think>')
                                if think_end != -1:
                                    # skip thinking content, continue after tag
                                    buffer = buffer[think_end + 8:]  # skip '</think>'
                                    in_thinking = False
                                else:
                                    # still in thinking, discard buffer but keep tail
                                    if len(buffer) > 10:
                                        buffer = buffer[-10:]
                                    break
                    elif isinstance(chunk, AIMsg) and chunk.content and not emitted_chunks:
                        # non-streaming fallback: emit once if no chunks were emitted
                        yield ChunkEvent(chunk.content)
                        emitted_chunks = True
                        
            if event_type == 'updates':
                if "_retrieve" in event_data:
                    payload = event_data["_retrieve"]
                    documents = payload["context"]

                    unique_docs = []
                    seen_content = set()
                    for doc in documents:
                        if doc.page_content not in seen_content:
                            unique_docs.append(doc)
                            seen_content.add(doc.page_content)
                    yield SourcesEvent(unique_docs)

                    confidence = payload.get("confidence")
                    if confidence:
                        yield ConfidenceEvent(confidence)

        # flush remaining buffer (content after thinking or remaining text)
        if buffer and not in_thinking:
            yield ChunkEvent(buffer)

    def ask(self, prompt: str, chat_history: List[Message]) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        for event in self._ask_model(prompt, chat_history):
            yield event