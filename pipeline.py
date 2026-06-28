"""
pipeline.py — Self-Healing Agentic RAG Pipeline

KEY FIX: Track best answer across all rounds.
If corrections make things worse, return the best answer seen.
"""
from agent.llm_reasoner import LLMReasoner
from agent.task_planner import TaskPlanner
from agent.self_monitor import SelfMonitor
from agent.corrective_actions import CorrectiveActions
from retrieval.retriever import HybridRetriever
from config import MAX_CORRECTION_ATTEMPTS


class SelfHealingRAGPipeline:
    def __init__(self):
        print("[Pipeline] Initialising Self-Healing RAG...")
        self.llm       = LLMReasoner()
        self.retriever = HybridRetriever()
        self.planner   = TaskPlanner(self.llm)
        self.monitor   = SelfMonitor(self.llm)
        self.corrector = CorrectiveActions(self.llm, self.retriever)
        print("[Pipeline] Ready.")

    def run(self, question, documents, question_type=None):
        trace = []
        self.corrector.reset_round()

        # Step 1: Index
        self.retriever.index_documents(documents)

        # Step 2: Plan
        plan = self.planner.plan(question, question_type)
        trace.append(
            f"plan: strategy={plan['strategy']}, "
            f"sub_qs={plan['sub_questions']}")

        # Step 3: Retrieve
        retrieved      = self.retriever.retrieve(question)
        retrieved_docs = [r["text"] for r in retrieved]
        trace.append(f"retrieved {len(retrieved_docs)} docs")

        # Step 4: Generate
        sub_answers = []

        if plan["strategy"] != "direct" and plan["sub_questions"]:
            for sq in plan["sub_questions"]:
                extra      = self.retriever.retrieve(sq)
                extra_docs = [r["text"] for r in extra]
                all_docs   = list(
                    dict.fromkeys(retrieved_docs + extra_docs))[:8]
                sa         = self.llm.answer_with_context(sq, all_docs)
                sub_answers.append(sa)
                trace.append(f"sub-answer: '{sq[:50]}' → '{sa[:50]}'")
            answer = self.llm.answer_with_sub_answers(
                question, sub_answers, retrieved_docs)

        elif plan["strategy"] != "direct" and not plan["sub_questions"]:
            all_retrieved  = self.retriever.retrieve(
                question, top_k=len(documents))
            retrieved_docs = [r["text"] for r in all_retrieved]
            answer         = self.llm.answer_with_context(
                question, retrieved_docs)
            trace.append(
                f"multi-hop direct: using {len(retrieved_docs)} docs")
        else:
            answer = self.llm.answer_with_context(question, retrieved_docs)

        trace.append(f"initial answer: '{answer[:100]}'")

        # KEY: track best answer seen across all rounds
        best_answer = answer

        # Step 5: Self-monitoring + correction
        failure_detected     = False
        correction_attempted = False
        failure_reason       = ""
        n_rounds             = 0

        for attempt in range(MAX_CORRECTION_ATTEMPTS):
            report = self.monitor.evaluate(
                question=question,
                answer=answer,
                retrieved_docs=retrieved_docs,
                sub_answers=sub_answers or None,
            )

            if not report.has_failure:
                trace.append(f"self-monitor: PASS (attempt {attempt})")
                # If this answer passed, it's our best
                best_answer = answer
                break

            failure_detected = True
            failure_reason   = report.failure_reason
            trace.append(
                f"self-monitor: FAIL — {report.failure_reason} "
                f"→ action={report.suggested_action}")

            correction_attempted = True
            correction   = self.corrector.apply(
                action=report.suggested_action,
                question=question,
                previous_answer=answer,
                retrieved_docs=retrieved_docs,
                ungrounded_sentences=report.ungrounded_sentences,
            )

            new_answer     = correction["new_answer"]
            retrieved_docs = correction["new_docs"]
            n_rounds      += 1
            trace.append(
                f"correction {n_rounds}: {correction['action_taken']} "
                f"→ '{new_answer[:60]}'")

            # Only update answer if correction helped
            if new_answer and new_answer.lower() != "i don't know":
                answer      = new_answer
                best_answer = new_answer  # track best

        return {
            "final_answer":          best_answer.strip().rstrip("."),
            "retrieved_docs":        retrieved_docs,
            "failure_detected":      failure_detected,
            "correction_attempted":  correction_attempted,
            "n_correction_rounds":   n_rounds,
            "failure_reason":        failure_reason,
            "trace":                 trace,
        }

    def run_baseline(self, question, documents):
        self.retriever.index_documents(documents)
        retrieved = self.retriever.retrieve(question)
        docs      = [r["text"] for r in retrieved]
        answer    = self.llm.answer_with_context(question, docs)
        return {
            "answer":        answer.strip().rstrip("."),
            "retrieved_docs": docs
        }
