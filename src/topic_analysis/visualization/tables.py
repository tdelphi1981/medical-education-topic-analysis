"""LaTeX table generation utilities."""

from pathlib import Path
from typing import Optional

from ..config import LATEX_OUTPUT_DIR


def generate_topic_table(
    topics: list[dict],
    model_name: str,
    num_words: int = 10,
    output_path: Optional[Path] = None,
) -> str:
    """Generate LaTeX table displaying topics.

    Args:
        topics: List of topic dictionaries with 'id' and 'words' keys.
        model_name: Name of the model for the table caption.
        num_words: Number of words to show per topic.
        output_path: Path to save the table.

    Returns:
        LaTeX table code.
    """
    rows = []
    for topic in topics:
        topic_id = topic.get("id", topic.get("topic_id", "?"))
        words = topic.get("words", [])[:num_words]
        words_str = ", ".join(words)
        rows.append(f"{topic_id} & {words_str} \\\\")

    table = r"""
\begin{table}[htbp]
\centering
\caption{""" + model_name + r""" topics (top-""" + str(num_words) + r""" words)}
\label{tab:""" + model_name.lower().replace(" ", "-") + r"""-topics}
\begin{tabular}{cp{10cm}}
\toprule
\textbf{Topic} & \textbf{Top Words} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)

    return table


def generate_results_table(
    results: dict[str, dict],
    topic_counts: list[int],
    metric: str = "c_v",
    output_path: Optional[Path] = None,
) -> str:
    """Generate LaTeX table with coherence results across topic counts.

    Args:
        results: Nested dict {model_name: {topic_count: metrics}}.
        topic_counts: List of topic counts.
        metric: Which coherence metric to display.
        output_path: Path to save the table.

    Returns:
        LaTeX table code.
    """
    # Header with topic counts
    header = " & ".join([f"\\textbf{{K={k}}}" for k in topic_counts])

    rows = []
    for model_name, model_results in results.items():
        values = []
        for k in topic_counts:
            if k in model_results:
                score = model_results[k].get("coherence", {}).get(metric, "N/A")
                if isinstance(score, float):
                    values.append(f"{score:.4f}")
                else:
                    values.append("--")
            else:
                values.append("--")
        rows.append(f"{model_name} & " + " & ".join(values) + " \\\\")

    table = r"""
\begin{table}[htbp]
\centering
\caption{""" + metric.replace("_", r"\_") + r""" coherence scores by method and number of topics}
\label{tab:""" + metric + r"""-coherence}
\begin{tabular}{l""" + "c" * len(topic_counts) + r"""}
\toprule
\textbf{Method} & """ + header + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)

    return table


def generate_diversity_table(
    results: dict[str, dict],
    topic_counts: list[int],
    output_path: Optional[Path] = None,
) -> str:
    """Generate LaTeX table with diversity scores.

    Args:
        results: Nested dict {model_name: {topic_count: metrics}}.
        topic_counts: List of topic counts.
        output_path: Path to save the table.

    Returns:
        LaTeX table code.
    """
    header = " & ".join([f"\\textbf{{K={k}}}" for k in topic_counts])

    rows = []
    for model_name, model_results in results.items():
        values = []
        for k in topic_counts:
            if k in model_results:
                score = model_results[k].get("diversity", "N/A")
                if isinstance(score, float):
                    values.append(f"{score:.4f}")
                else:
                    values.append("--")
            else:
                values.append("--")
        rows.append(f"{model_name} & " + " & ".join(values) + " \\\\")

    table = r"""
\begin{table}[htbp]
\centering
\caption{Topic diversity scores (proportion of unique words in top-10)}
\label{tab:diversity}
\begin{tabular}{l""" + "c" * len(topic_counts) + r"""}
\toprule
\textbf{Method} & """ + header + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)

    return table
