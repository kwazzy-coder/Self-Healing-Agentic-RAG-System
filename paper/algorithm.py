"""
paper/algorithm.py — LaTeX Algorithm Generation

Generates publication-ready pseudocode for the paper.
"""

def generate_main_algorithm() -> str:
    return r"""
\begin{algorithm}[t]
\caption{ETSC-RAG: Efficient Tiered Self-Correcting RAG}
\label{alg:etsc}
\begin{algorithmic}[1]
\Require Query $q$, Document corpus $\mathcal{D}$, Max rounds $R$
\Ensure Answer $a$
\State $\mathcal{D}_{\text{ret}} \gets \textsc{HybridRetrieve}(q, \mathcal{D})$ \Comment{BM25 + Dense fusion}
\State $\textit{plan} \gets \textsc{TaskPlanner}(q)$ \Comment{Detect query complexity}
\If{$\textit{plan.strategy} \neq \texttt{direct}$}
    \State $\{q_1, q_2\} \gets \textsc{Decompose}(q, \textit{plan})$
    \State $a \gets \textsc{SynthesizeSubs}(q_1, q_2, \mathcal{D}_{\text{ret}})$
\Else
    \State $a \gets \textsc{Generate}(q, \mathcal{D}_{\text{ret}})$ \Comment{CoT prompting}
\EndIf
\For{$r = 1$ \textbf{to} $R$}
    \State $(\textit{tier}, \textit{fail}) \gets \textsc{TieredMonitor}(q, a, \mathcal{D}_{\text{ret}})$
    \If{$\textit{fail} = \emptyset$}
        \State \Return $a$ \Comment{Answer passed all checks}
    \EndIf
    \State $\textit{corr} \gets \textsc{AdaptiveCorrect}(\textit{fail.type}, q, a, \mathcal{D}_{\text{ret}})$
    \State $a \gets \textit{corr.answer}$;\ $\mathcal{D}_{\text{ret}} \gets \mathcal{D}_{\text{ret}} \cup \textit{corr.docs}$
\EndFor
\State \Return $a$
\end{algorithmic}
\end{algorithm}
"""

def generate_monitor_algorithm() -> str:
    return r"""
\begin{algorithm}[t]
\caption{TieredMonitor: Cost-Efficient Failure Detection}
\label{alg:monitor}
\begin{algorithmic}[1]
\Require Query $q$, Answer $a$, Context $\mathcal{D}$, Thresholds $\tau_l, \tau_h$
\Ensure (tier, failure\_report)
\Statex \textit{// Tier 1: Zero-cost heuristics (no model calls)}
\If{$\textsc{IsEmpty}(a)$ \textbf{or} $\textsc{IsRefusal}(a)$}
    \State \Return $(1, \texttt{REFUSAL})$
\EndIf
\If{$\textsc{KeywordOverlap}(a, q) = 0$}
    \State \Return $(1, \texttt{IRRELEVANT})$
\EndIf
\Statex \textit{// Tier 2: Lightweight NLI (67M params, no API call)}
\State $s_{\text{nli}} \gets \textsc{NLI}(\mathcal{D}, a)$ \Comment{Cross-encoder inference}
\If{$s_{\text{nli}} < \tau_l$}
    \State \Return $(2, \texttt{UNGROUNDED})$
\EndIf
\If{$s_{\text{nli}} > \tau_h$}
    \State \Return $(2, \emptyset)$ \Comment{High confidence pass}
\EndIf
\Statex \textit{// Tier 3: LLM-as-judge (1 API call, only when uncertain)}
\State $j \gets \textsc{LLMJudge}(q, a, \mathcal{D})$
\State \Return $(3, j)$
\end{algorithmic}
\end{algorithm}
"""

def save_algorithms(output_dir: str = "paper"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "algorithm1_etsc.tex"), "w") as f:
        f.write(generate_main_algorithm())
    with open(os.path.join(output_dir, "algorithm2_monitor.tex"), "w") as f:
        f.write(generate_monitor_algorithm())
    print(f"Algorithms saved to {output_dir}/")

if __name__ == "__main__":
    save_algorithms()
