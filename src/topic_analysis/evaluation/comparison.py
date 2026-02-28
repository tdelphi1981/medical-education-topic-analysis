"""Model comparison utilities."""

from typing import Optional
import json
from pathlib import Path

from .coherence import compute_all_coherence_metrics
from .diversity import compute_topic_diversity
from ..config import EVALUATION_CONFIG, RESULTS_DIR


def compare_models(
    model_results: dict[str, dict],
    texts: list[list[str]],
    dictionary,
    save_path: Optional[Path] = None,
) -> dict:
    """Compare multiple topic models across evaluation metrics.

    Args:
        model_results: Dictionary mapping model names to their results.
            Each result should have a "topics" key with list of word lists.
        texts: Tokenized corpus for coherence computation.
        dictionary: Gensim dictionary.
        save_path: Path to save comparison results.

    Returns:
        Dictionary with comparison metrics for each model.
    """
    comparison = {}

    for model_name, results in model_results.items():
        topics = results.get("topics", [])

        if not topics:
            print(f"Warning: No topics found for {model_name}")
            continue

        # Extract word lists
        if isinstance(topics[0], dict):
            topic_words = [t["words"] for t in topics]
        else:
            topic_words = topics

        # Compute metrics
        coherence_scores = compute_all_coherence_metrics(
            topic_words, texts, dictionary
        )
        diversity = compute_topic_diversity(topic_words)

        comparison[model_name] = {
            "num_topics": len(topics),
            "coherence": coherence_scores,
            "diversity": diversity,
        }

        print(f"\n{model_name}:")
        print(f"  Topics: {len(topics)}")
        print(f"  Diversity: {diversity:.4f}")
        for measure, score in coherence_scores.items():
            if score is not None:
                print(f"  {measure}: {score:.4f}")

    # Save results
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(comparison, f, indent=2)

    return comparison


def generate_comparison_table(comparison: dict, metric: str = "c_v") -> str:
    """Generate LaTeX table comparing models.

    Args:
        comparison: Comparison results dictionary.
        metric: Coherence metric to highlight.

    Returns:
        LaTeX table code.
    """
    rows = []
    for model_name, metrics in comparison.items():
        coherence = metrics["coherence"].get(metric, "N/A")
        coherence_str = f"{coherence:.4f}" if isinstance(coherence, float) else str(coherence)
        diversity = f"{metrics['diversity']:.4f}"
        num_topics = metrics["num_topics"]

        rows.append(f"{model_name} & {num_topics} & {coherence_str} & {diversity} \\\\")

    table = r"""
\begin{table}[htbp]
\centering
\caption{Topic model comparison}
\label{tab:model-comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Topics} & \textbf{""" + metric.replace("_", r"\_") + r"""} & \textbf{Diversity} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table
