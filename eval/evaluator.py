import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from langchain_core.documents import Document

from chatbot import Role

@dataclass
class RetrievalResult:
    """
    Result of a single retrieval operation
    """
    question_id: str
    question: str
    retrieved_docs: List[Document]
    latency_ms: float
    hit: bool = False
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0

@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics"""
    total_questions: int = 0
    hit_rate: float = 0.0
    mean_recall_at_k: float = 0.0
    mrr: float = 0.0  # mean reciprocal rank
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    # breakdown by content type
    table_hit_rate: float = 0.0
    text_hit_rate: float = 0.0

@dataclass 
class EvalConfig:
    """
    Configuration for evaluation run
    """
    k: int = 4
    gold_set_path: Path = Path('eval/gold_set.json')
    output_dir: Path = Path('eval/results')

class Evaluator: 
    """
    Evaluates retrieval quality against a gold set
    """
    def __init__(self, retriever, config: EvalConfig = None):
        self.retriever = retriever
        self.config = config or EvalConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_gold_set(self) -> List[Dict]:
        """Load gold set from JSON file"""
        with open(self.config.gold_set_path, 'r') as f:
            data = json.load(f)
        return data.get('questions', [])
    
    def _check_hit(self, retrieved_docs: List[Document], gold: Dict) -> bool:
        """
        Check if any retrieved doc matches expected criteria
        """
        expected_sources = set(gold.get('expected_sources', []))
        expected_pages = set(gold.get('expected_pages', []))
        keywords = gold.get('keywords', [])
        
        for doc in retrieved_docs:
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page')
            content = doc.page_content.lower()
            
            # check source match
            source_match = not expected_sources or any(
                exp in source for exp in expected_sources
            )
            
            # check page match (if specified)
            page_match = not expected_pages or page in expected_pages
            
            # check keyword match (at least one keyword present)
            keyword_match = not keywords or any(
                kw.lower() in content for kw in keywords
            )
            
            if source_match and page_match and keyword_match:
                return True
        
        return False

    def _compute_recall_at_k(
        self, retrieved_docs: List[Document], gold: Dict, k: int
    ) -> float:
        """
        Compute recall@k: fraction of expected items found in top k
        """
        expected_pages = set(gold.get('expected_pages', []))
        if not expected_pages:
            return 1.0 if self._check_hit(retrieved_docs[:k], gold) else 0.0

        found_pages = set()
        for doc in retrieved_docs[:k]:
            page = doc.metadata.get('page')
            if page in expected_pages:
                found_pages.add(page)

        return len(found_pages)/len(expected_pages)

    def _compute_reciprocal_rank(
        self, retrieval_docs: List[Document], gold: Dict
    ) -> float:
        """
        Compute: Reciprocal Rank: 1/rank of first relevant document
        """
        for i, doc in enumerate(retrieval_docs):
            if self._check_hit([doc], gold):
                return 1.0/(i+1)
        return 0.0

    def evaluate_single(self, gold_item: Dict) -> RetrievalResult:
        """
        Evaluate a single question
        """
        question = gold_item['question']
        question_id = gold_item.get('id', question[:20])
        start = time.perf_counter()
        retrieved_docs = self.retriever.invoke(question)
        latency_ms = (time.perf_counter() - start) * 1000
        # compute metrics
        hit = self._check_hit(retrieved_docs[:self.config.k], gold_item)
        recall = self._compute_recall_at_k(retrieved_docs, gold_item, self.config.k)
        rr = self._compute_reciprocal_rank(retrieved_docs, gold_item)

        return RetrievalResult(
            question_id=question_id,
            question=question,
            retrieved_docs=retrieved_docs,
            latency_ms=latency_ms,
            hit=hit,
            recall_at_k=recall,
            reciprocal_rank=rr
        )

    def run(self) -> tuple[List[RetrievalResult], EvalMetrics]:
        """Run evaluation on entire gold set"""
        gold_set = self.load_gold_set()
        results: List[RetrievalResult] = []
        
        print(f"🧪 Running evaluation on {len(gold_set)} questions (k={self.config.k})...")
        
        for i, gold_item in enumerate(gold_set):
            print(f"  [{i+1}/{len(gold_set)}] {gold_item['question'][:50]}...")
            result = self.evaluate_single(gold_item)
            results.append(result)
            
            status = "✅" if result.hit else "❌"
            print(f"    {status} Hit={result.hit}, Recall@{self.config.k}={result.recall_at_k:.2f}, Latency={result.latency_ms:.0f}ms")
        
        # aggregate metrics
        metrics = self._aggregate_metrics(results, gold_set)
        
        return results, metrics
    
    def _aggregate_metrics(
        self, results: List[RetrievalResult], gold_set: List[Dict]
    ) -> EvalMetrics:
        """
        Compute aggregate metrics from individual results
        """
        if not results:
            return EvalMetrics()
        
        latencies = [r.latency_ms for r in results]
        latencies_sorted = sorted(latencies)
        
        # breakdown by content type
        table_results = [
            r for r, g in zip(results, gold_set) 
            if g.get('expected_content_type') == 'table'
        ]
        text_results = [
            r for r, g in zip(results, gold_set) 
            if g.get('expected_content_type') == 'text'
        ]
        
        return EvalMetrics(
            total_questions=len(results),
            hit_rate=sum(r.hit for r in results) / len(results),
            mean_recall_at_k=sum(r.recall_at_k for r in results) / len(results),
            mrr=sum(r.reciprocal_rank for r in results) / len(results),
            mean_latency_ms=sum(latencies) / len(latencies),
            p95_latency_ms=latencies_sorted[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            table_hit_rate=sum(r.hit for r in table_results) / len(table_results) if table_results else 0.0,
            text_hit_rate=sum(r.hit for r in text_results) / len(text_results) if text_results else 0.0,
        )

    def evaluate_with_faithfulness(
        self,
        chatbot,  
        verbose: bool = False
        ) -> Tuple[List[dict], dict]:
        """
        Complete evaluation: retrieval + generation + faithfulness.
        
        Measures:
        1. Retrieval quality (Recall@k, MRR, Hit Rate)
        2. Answer faithfulness (LLM-as-Judge)
        3. End-to-end latency
        """
        from eval.llm_judge import LLMJudge
        from chatbot import SourcesEvent, FinalAnswerEvent, ChunkEvent, Message
        from config import Config
        import time
        
        gold_set = self.load_gold_set()
        judge = LLMJudge(
            model_name=Config.Eval.JUDGE_MODEL,
            verbose=verbose
        )
        
        detailed_results = []
        
        print(f"\n{'='*60}")
        print(f"FULL RAG EVALUATION")
        print(f"Questions: {len(gold_set)}")
        print(f"Retrieval k: {self.config.k}")
        print(f"Judge model: {Config.Eval.JUDGE_MODEL}")
        print(f"{'='*60}\n")
        
        for i, gold_item in enumerate(gold_set):
            question = gold_item['question']
            print(f"[{i+1}/{len(gold_set)}] {question[:60]}...")
            
            # collect events from chatbot
            context_docs = []
            answer_chunks = []
            start_time = time.perf_counter()
            
            try:
                print("  Generating: ", end="", flush=True)
                for event in chatbot.ask(question, [Message(role=Role.USER, content="")]):
                    if isinstance(event, SourcesEvent):
                        context_docs = event.content
                    elif isinstance(event, ChunkEvent):
                        answer_chunks.append(event.content)
                        print(event.content, end="", flush=True)  # Stream live
                    elif isinstance(event, FinalAnswerEvent):
                        # some systems emit final answer event
                        pass
                print()  # Newline after streaming completes
                
                answer = "".join(answer_chunks).strip()
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # retrieval metrics
                hit = self._check_hit(context_docs[:self.config.k], gold_item)
                recall = self._compute_recall_at_k(context_docs, gold_item, self.config.k)
                rr = self._compute_reciprocal_rank(context_docs, gold_item)
                
                # faithfulness evaluation
                if answer and len(answer) > 10:  # only judge substantial answers
                    judgment = judge.evaluate(question, context_docs, answer)
                    
                    result = {
                        "question": question,
                        "question_id": gold_item.get('id', f"q{i+1}"),
                        "answer": answer,
                        "answer_length": len(answer),
                        
                        # retrieval metrics
                        "retrieval_hit": hit,
                        "recall_at_k": recall,
                        "reciprocal_rank": rr,
                        "num_sources": len(context_docs),
                        
                        # faithfulness metrics
                        "faithfulness": judgment.faithfulness_score,
                        "relevance": judgment.relevance_score,
                        "completeness": judgment.completeness_score,
                        "overall_quality": judgment.overall_score,
                        "hallucination_detected": judgment.hallucination_detected,
                        "judge_reasoning": judgment.reasoning,
                        
                        # performance
                        "latency_ms": latency_ms,
                    }
                    
                    # status indicator
                    hit_emoji = "✅" if hit else "❌"
                    faith_emoji = "✓" if judgment.faithfulness_score >= 0.7 else "~" if judgment.faithfulness_score >= 0.4 else "✗"
                    halluc_flag = " [HALLUC]" if judgment.hallucination_detected else ""
                    
                    print(f"  {hit_emoji} Hit={hit} | Recall@{self.config.k}={recall:.2f} | RR={rr:.2f}")
                    print(f"  {faith_emoji} Faith={judgment.faithfulness_score:.2f} Rel={judgment.relevance_score:.2f} Compl={judgment.completeness_score:.2f}{halluc_flag}")
                    # Print generated answer (truncated)
                    answer_preview = (answer[:500] + "...") if len(answer) > 500 else answer
                    print(f"  📝 Answer: {answer_preview}")
                    if verbose:
                        print(f"    Reason: {judgment.reasoning[:100]}")
                
                else:
                    result = {
                        "question": question,
                        "question_id": gold_item.get('id', f"q{i+1}"),
                        "answer": answer or "(empty)",
                        "retrieval_hit": hit,
                        "recall_at_k": recall,
                        "reciprocal_rank": rr,
                        "faithfulness": 0.0,
                        "relevance": 0.0,
                        "hallucination_detected": False,
                        "judge_reasoning": "Answer too short or empty",
                        "latency_ms": latency_ms,
                    }
                    print(f"Empty or short answer")
                
                detailed_results.append(result)
                
            except Exception as e:
                print(f"Error: {e}")
                detailed_results.append({
                    "question": question,
                    "question_id": gold_item.get('id', f"q{i+1}"),
                    "error": str(e),
                })
        
        # aggregate metrics
        valid_results = [r for r in detailed_results if "error" not in r]
        n = len(valid_results)
        
        if n == 0:
            print("\nAll evaluations failed")
            return detailed_results, {}
        
        aggregate = {
            "total_questions": len(gold_set),
            "successful_evaluations": n,
            
            # retrieval
            "retrieval_hit_rate": sum(r["retrieval_hit"] for r in valid_results) / n,
            "mean_recall_at_k": sum(r["recall_at_k"] for r in valid_results) / n,
            "mean_reciprocal_rank": sum(r.get("reciprocal_rank", 0) for r in valid_results) / n,
            
            # faithfulness
            "mean_faithfulness": sum(r.get("faithfulness", 0) for r in valid_results) / n,
            "mean_relevance": sum(r.get("relevance", 0) for r in valid_results) / n,
            "mean_completeness": sum(r.get("completeness", 0) for r in valid_results) / n,
            "mean_overall_quality": sum(r.get("overall_quality", 0) for r in valid_results) / n,
            
            # safety
            "hallucination_rate": sum(r.get("hallucination_detected", False) for r in valid_results) / n,
            "answers_below_faith_threshold": sum(
                1 for r in valid_results 
                if r.get("faithfulness", 1) < Config.Eval.FAITHFULNESS_THRESHOLD
            ) / n,
            
            # performance
            "mean_latency_ms": sum(r["latency_ms"] for r in valid_results) / n,
        }
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Retrieval Hit Rate:     {aggregate['retrieval_hit_rate']:.1%}")
        print(f"Mean Recall@{self.config.k}:         {aggregate['mean_recall_at_k']:.1%}")
        print(f"MRR:                    {aggregate['mean_reciprocal_rank']:.3f}")
        print(f"---")
        print(f"Mean Faithfulness:      {aggregate['mean_faithfulness']:.1%}")
        print(f"Mean Relevance:         {aggregate['mean_relevance']:.1%}")
        print(f"Overall Quality:        {aggregate['mean_overall_quality']:.1%}")
        print(f"---")
        print(f"Hallucination Rate:     {aggregate['hallucination_rate']:.1%}")
        print(f"Below Threshold:        {aggregate['answers_below_faith_threshold']:.1%}")
        print(f"---")
        print(f"Avg Latency:            {aggregate['mean_latency_ms']:.0f}ms")
        print(f"{'='*60}\n")
        
        # save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.config.output_dir / f"eval_detailed_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                "aggregate": aggregate,
                "detailed": detailed_results,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {results_path}\n")
        
        return detailed_results, aggregate
    
    def generate_report(
        self, results: List[RetrievalResult], metrics: EvalMetrics
    ) -> str:
        """Generate markdown evaluation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        report = f"""# Private-RAG Evaluation Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Questions**: {metrics.total_questions}  
**k**: {self.config.k}

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| **Hit Rate** | {metrics.hit_rate:.1%} |
| **Mean Recall@{self.config.k}** | {metrics.mean_recall_at_k:.1%} |
| **MRR** | {metrics.mrr:.3f} |
| **Mean Latency** | {metrics.mean_latency_ms:.0f} ms |
| **P95 Latency** | {metrics.p95_latency_ms:.0f} ms |

### By Content Type

| Type | Hit Rate |
|------|----------|
| Tables | {metrics.table_hit_rate:.1%} |
| Text | {metrics.text_hit_rate:.1%} |

---

## Per-Question Results

| ID | Question | Hit | Recall@{self.config.k} | RR | Latency |
|----|----------|-----|----------|-----|---------|
"""
        for r in results:
            status = "✅" if r.hit else "❌"
            report += f"| {r.question_id} | {r.question[:40]}... | {status} | {r.recall_at_k:.2f} | {r.reciprocal_rank:.2f} | {r.latency_ms:.0f}ms |\n"
        
        report += """
---

## Interpretation

- **Hit Rate**: % of queries where at least one relevant doc was in top-k
- **Recall@k**: Average fraction of expected pages/sources found in top-k  
- **MRR**: Mean Reciprocal Rank — how high the first relevant result ranks
- **Latency**: End-to-end retrieval time (embedding + search + rerank)

## Recommendations

"""
        if metrics.hit_rate < 0.7:
            report += "- ⚠️ Hit rate below 70% — consider increasing k or tuning retrieval weights\n"
        if metrics.table_hit_rate < metrics.text_hit_rate - 0.2:
            report += "- ⚠️ Table retrieval underperforming text — check table chunking/BM25 text\n"
        if metrics.mean_latency_ms > 1000:
            report += "- ⚠️ High latency — consider caching or reducing reranker candidates\n"
        if metrics.hit_rate >= 0.8 and metrics.mrr >= 0.6:
            report += "- ✅ Retrieval quality looks good!\n"
        
        # save report
        report_path = self.config.output_dir / f"eval_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📊 Report saved to: {report_path}")

def run_evaluation(retriever, gold_set_path: str = "eval/gold_set.json", k: int = 4):
    """
    Convenience function to run evaluation.
    
    Usage:
        from eval.evaluator import run_evaluation
        from data_ingestor import ingest_files
        
        retriever = ingest_files(files)
        run_evaluation(retriever, k=4)
    """
    config = EvalConfig(k=k, gold_set_path=Path(gold_set_path))
    evaluator = Evaluator(retriever, config)
    results, metrics = evaluator.run()
    report = evaluator.generate_report(results, metrics)
    
    print("\n" + "="*50)
    print(f"📈 FINAL RESULTS")
    print(f"   Hit Rate:    {metrics.hit_rate:.1%}")
    print(f"   Recall@{k}:   {metrics.mean_recall_at_k:.1%}")
    print(f"   MRR:         {metrics.mrr:.3f}")
    print(f"   Avg Latency: {metrics.mean_latency_ms:.0f}ms")
    print("="*50)
    
    return results, metrics

    
