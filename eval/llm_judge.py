"""
LLM-as-Judge: Evaluate RAG response quality.

Evaluates three dimensions:
1. Faithfulness: Is the answer grounded in the retrieved context?
2. Relevance: Does the answer address the question?
3. Completeness: Does the answer use all relevant information?

Based on research:
- "FActScore" (Min et al. 2023)
- "SelfCheckGPT" (Manakul et al. 2023)
- Production practices from Cohere, Anthropic

Usage:
    judge = LLMJudge()
    result = judge.evaluate(question, context_docs, answer)
    print(f"Faithfulness: {result.faithfulness_score:.2f}")
    print(f"Hallucination: {result.hallucination_detected}")
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llama_wrapper import ChatLlamaCppWrapper
from thinking_utils import strip_thinking_tags


@dataclass
class JudgmentResult:
    """Result from LLM judge evaluation"""
    faithfulness_score: float  # 0-1: answer grounded in context?
    relevance_score: float     # 0-1: answer addresses question?
    completeness_score: float  # 0-1: uses all relevant info?
    overall_score: float       # average of above
    reasoning: str             # judge's explanation
    hallucination_detected: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "faithfulness": self.faithfulness_score,
            "relevance": self.relevance_score,
            "completeness": self.completeness_score,
            "overall": self.overall_score,
            "reasoning": self.reasoning,
            "hallucination_detected": self.hallucination_detected,
        }

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a STRICT evaluator for RAG (Retrieval-Augmented Generation) systems.

Your task: Assess if the ANSWER is faithful to the CONTEXT (no hallucinations).

IMPORTANT: Be SKEPTICAL. Most answers are NOT perfect. Score critically.

EVALUATION CRITERIA

**1. FAITHFULNESS (0-10)**
Every claim in the answer must be EXPLICITLY stated in the context.

STRICT Scoring Guide:
• 10: RARE - Every single word/number has explicit verbatim support. No inference.
• 8-9: Very Good - All key facts correct, tiny stylistic additions only
• 6-7: Good - Main facts correct, but some minor details not in context
• 4-5: Mixed - Correct answer but includes unsupported claims
• 2-3: Poor - Significant fabrication or wrong numbers/names
• 0-1: Failed - Wrong answer, contradicts context, or complete hallucination

RED FLAGS (lower score immediately):
- Numbers not exactly matching context
- Adding context/details not in the source
- Confident claims about things not mentioned
- Rounding or paraphrasing numbers differently

**2. RELEVANCE (0-10)**
Does the answer actually address what was asked?

Scoring Guide:
• 10: Directly and completely answers the question
• 7-9: Addresses main points, might miss minor aspects
• 4-6: Partially relevant, includes tangential information
• 1-3: Barely addresses question
• 0: Completely off-topic

**3. COMPLETENESS (0-10)**
Does the answer use all relevant information from context?

Scoring Guide:
• 10: Uses all relevant context appropriately
• 7-9: Misses some relevant details
• 4-6: Ignores significant relevant information
• 1-3: Uses very little of the available context
• 0: Ignores all relevant context

**4. HALLUCINATION DETECTION**
Set to TRUE if:
- Answer makes specific claims (numbers, names, facts) not in context
- Answer contradicts information in the context
- Answer fabricates details

Set to FALSE if:
- All claims are supported or are reasonable inferences
- Minor paraphrasing or summarization

OUTPUT FORMAT

Respond ONLY with valid JSON (no markdown, no explanation):
json
{{
"faithfulness": <0-10>,
"relevance": <0-10>,
"completeness": <0-10>,
"hallucination_detected": <true/false>,
"reasoning": "<1-2 sentence explanation>"
}}


EXAMPLES

Question: "What is the capital of France?"
Context: "Paris is the capital and largest city of France."
Answer: "Paris"
Output: {{"faithfulness": 10, "relevance": 10, "completeness": 10, "hallucination_detected": false, "reasoning": "Direct match with context, fully grounded."}}

Question: "What is the capital of France?"
Context: "Paris is the capital and largest city of France."
Answer: "Paris, with a population of 2.2 million people."
Output: {{"faithfulness": 5, "relevance": 9, "completeness": 7, "hallucination_detected": true, "reasoning": "Answer includes population not mentioned in context."}}

Question: "What is the capital of France?"
Context: "Lyon is a major city in France."
Answer: "Paris"
Output: {{"faithfulness": 0, "relevance": 10, "completeness": 0, "hallucination_detected": true, "reasoning": "Answer makes claim not supported by provided context."}}


Now evaluate the following:
"""),
    ("human", """QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Your evaluation (JSON only):"""),
])


class LLMJudge:
    """
    LLM-based judge for evaluating RAG response quality
    """
    def __init__(
        self, model_name: str = 'qwen3:4b-thinking',
        temperature: float = 0.0,
        verbose: bool = False,
        llm = None  # Optional: pass existing LLM instance to reuse
    ):
        """
        Initialize LLM Judge.
        
        Args:
            llm: Optional existing LLM to reuse (e.g., from chatbot).
                 If None, uses the shared chatbot LLM singleton.
        """
        self.model_name = model_name
        self.verbose = verbose

        # Reuse the shared chatbot LLM singleton to save GPU memory
        if llm is not None:
            self.llm = llm
            if verbose:
                print(f'LLM Judge using provided LLM instance')
        else:
            from data_ingestor import get_chatbot_llm
            self.llm = get_chatbot_llm()
            if verbose:
                from config import Config
                print(f'LLM Judge reusing chatbot LLM: {Config.Model.GGUF_PATH.name}')

        self.chain = FAITHFULNESS_PROMPT | self.llm | StrOutputParser()

    def _format_context(self, docs: List[Document], max_chars: int = 3000) -> str:
        """
        Format retrieved documents for judge evaluation
        """
        formatted = []
        total_chars = 0
        for i, doc in enumerate(docs):
            # truncate individual doc
            content = doc.page_content[:1000]
            if len(doc.page_content) > 1000:
                content += "\n[...truncated...]"

            # add metadata for traceability
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            content_type = doc.metadata.get('content_type', 'text')
            entry = f"[Source {i+1}: {source}, Page {page}, Type: {content_type}]\n{content}"
            
            # stop if max_chars exceeded
            if total_chars + len(entry) > max_chars:
                formatted.append(f"\n[...{len(docs) - i} more sources omitted...]")
                break
            
            formatted.append(entry)
            total_chars += len(entry)
        
        return "\n\n---\n\n".join(formatted)

    def _parse_judgment(self, raw_output: str) -> JudgmentResult:
        """
        Parse LLM judge output into structured result.
        Handles:
        - Thinking tags (from some models)
        - Markdown code blocks
        - Malformed JSON
        - Missing fields
        """
        try:
            # CRITICAL: Strip thinking tags using robust utility
            raw_output = strip_thinking_tags(raw_output, aggressive=True)

            # remove markdown code blocks
            raw_output = re.sub(r'\s*', '', raw_output)
            raw_output = re.sub(r'```\s*$', '', raw_output)
            
            # extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if not json_match:
                raise ValueError(f"No JSON found in response: {raw_output[:200]}")
            
            data = json.loads(json_match.group())
            
            # normalize scores from 0-10 to 0-1
            faithfulness = float(data.get('faithfulness', 5)) / 10.0
            relevance = float(data.get('relevance', 5)) / 10.0
            completeness = float(data.get('completeness', 5)) / 10.0
            
            # clamp to valid range
            faithfulness = max(0.0, min(1.0, faithfulness))
            relevance = max(0.0, min(1.0, relevance))
            completeness = max(0.0, min(1.0, completeness))
            
            hallucination = data.get('hallucination_detected', False)
            if isinstance(hallucination, str):
                hallucination = hallucination.lower() in ('true', 'yes', '1')
            
            reasoning = data.get('reasoning', 'No reasoning provided')
            
            result = JudgmentResult(
                faithfulness_score=faithfulness,
                relevance_score=relevance,
                completeness_score=completeness,
                overall_score=(faithfulness + relevance + completeness) / 3.0,
                reasoning=reasoning,
                hallucination_detected=bool(hallucination),
            )
            
            if self.verbose:
                print(f"✓ Parsed judgment: F={faithfulness:.2f}, R={relevance:.2f}, C={completeness:.2f}")
            
            return result

        except Exception as e:
            if self.verbose:
                print(f"Failed to parse judge output: {e}")
                print(f"Raw output: {raw_output[:200]}")
            
            # return neutral scores on parse failure
            return JudgmentResult(
                faithfulness_score=0.5,
                relevance_score=0.5,
                completeness_score=0.5,
                overall_score=0.5,
                reasoning=f"Parse error: {str(e)[:100]}",
                hallucination_detected=False,
            )

    def evaluate(
        self,
        question: str,
        context_docs: List[Document],
        answer: str
    ) -> JudgmentResult:
        """
        Evaluate a single response for faithfulness.
        """
        if not answer or not answer.strip():
            return JudgmentResult(
                faithfulness_score=0.0,
                relevance_score=0.0,
                completeness_score=0.0,
                overall_score=0.0,
                reasoning="Empty answer",
                hallucination_detected=False,
            )
        
        context_str = self._format_context(context_docs)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating answer to: {question[:60]}...")
            print(f"Context length: {len(context_str)} chars, {len(context_docs)} docs")
            print(f"Answer length: {len(answer)} chars")
        
        raw_output = self.chain.invoke({
            "question": question,
            "context": context_str,
            "answer": answer,
        })
        
        result = self._parse_judgment(raw_output)
        
        if self.verbose:
            status = "PASSED" if result.overall_score >= 0.7 else "ALMOST PASSED" if result.overall_score >= 0.4 else "FAILED"
            print(f"{status} Overall: {result.overall_score:.2f}, Halluc: {result.hallucination_detected}")
            print(f"{'='*60}\n")
        
        return result

    def evaluate_batch(
        self,
        items: List[Dict[str, any]]  # each: {"question": str, "context": List[Doc], "answer": str}
    ) -> List[JudgmentResult]:
        """Evaluate multiple responses (for bulk evaluation)"""
        results = []
        
        for i, item in enumerate(items):
            if self.verbose:
                print(f"\n[{i+1}/{len(items)}] Evaluating...")
            
            result = self.evaluate(
                item["question"],
                item["context"],
                item["answer"]
            )
            results.append(result)
        
        return results

def run_faithfulness_eval(
    questions: List[str],
    contexts: List[List[Document]],
    answers: List[str],
    model_name: str = "qwen3:4b-thinking",
    verbose: bool = False
) -> Dict[str, any]:
    """
    Convenience function to run faithfulness evaluation on a batch.
    
    Returns:
        dict with:
            - avg_faithfulness: mean faithfulness score
            - avg_relevance: mean relevance score
            - avg_overall: mean overall score
            - hallucination_rate: % of answers flagged as hallucinated
            - results: List[JudgmentResult]
    """
    judge = LLMJudge(model_name=model_name, verbose=verbose)
    
    results = []
    for q, c, a in zip(questions, contexts, answers):
        result = judge.evaluate(q, c, a)
        results.append(result)
    
    # Aggregate
    n = len(results)
    avg_faithfulness = sum(r.faithfulness_score for r in results) / n
    avg_relevance = sum(r.relevance_score for r in results) / n
    avg_completeness = sum(r.completeness_score for r in results) / n
    avg_overall = sum(r.overall_score for r in results) / n
    hallucination_rate = sum(r.hallucination_detected for r in results) / n
    
    return {
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "avg_completeness": avg_completeness,
        "avg_overall": avg_overall,
        "hallucination_rate": hallucination_rate,
        "total_evaluated": n,
        "results": results,
    }