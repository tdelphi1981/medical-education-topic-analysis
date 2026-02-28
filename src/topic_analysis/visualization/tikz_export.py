"""Export results as TikZ diagrams."""

from pathlib import Path
from typing import Optional

from ..config import LATEX_OUTPUT_DIR


def export_coherence_plot(
    results: dict[str, dict[str, list[float]]],
    topic_counts: list[int],
    metric: str = "c_v",
    output_path: Optional[Path] = None,
) -> str:
    """Export coherence comparison as TikZ/pgfplots code.

    Args:
        results: Dictionary mapping model names to {topic_count: score}.
        topic_counts: List of topic counts.
        metric: Coherence metric name.
        output_path: Path to save the TikZ file.

    Returns:
        TikZ code as string.
    """
    # Colors for each method
    colors = {
        "LDA": "ldacolor",
        "NMF": "nmfcolor",
        "BERTopic": "bertopiccolor",
    }

    # Generate plot coordinates
    plot_data = []
    for model_name, model_results in results.items():
        color = colors.get(model_name, "black")
        coords = " ".join([
            f"({k}, {v:.4f})" for k, v in sorted(model_results.items())
        ])
        plot_data.append(
            f"\\addplot[{color}, thick, mark=*] coordinates {{{coords}}};\n"
            f"\\addlegendentry{{{model_name}}}"
        )

    tikz_code = r"""% Coherence comparison plot
\begin{tikzpicture}
\begin{axis}[
    width=10cm,
    height=7cm,
    xlabel={Number of Topics},
    ylabel={""" + metric.replace("_", r"\_") + r""" Coherence},
    legend pos=south east,
    grid=major,
    xtick={""" + ",".join(map(str, topic_counts)) + r"""},
    ymin=0,
    ymax=1,
]
""" + "\n".join(plot_data) + r"""
\end{axis}
\end{tikzpicture}
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(tikz_code)

    return tikz_code


def export_topic_distribution(
    topic_proportions: list[float],
    topic_labels: list[str],
    output_path: Optional[Path] = None,
) -> str:
    """Export topic distribution as TikZ bar chart.

    Args:
        topic_proportions: List of topic proportions.
        topic_labels: List of topic labels.
        output_path: Path to save the TikZ file.

    Returns:
        TikZ code as string.
    """
    # Generate symbolic coordinates
    coords = " ".join([
        f"({label}, {prop:.4f})" for label, prop in zip(topic_labels, topic_proportions)
    ])

    tikz_code = r"""% Topic distribution bar chart
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=12cm,
    height=6cm,
    ylabel={Proportion},
    xlabel={Topic},
    symbolic x coords={""" + ",".join(topic_labels) + r"""},
    xtick=data,
    x tick label style={rotate=45, anchor=east},
    ymin=0,
    bar width=0.6cm,
    nodes near coords,
    nodes near coords style={font=\tiny},
]
\addplot[fill=nodeblue!60] coordinates {""" + coords + r"""};
\end{axis}
\end{tikzpicture}
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(tikz_code)

    return tikz_code


def export_year_topic_evolution(
    year_topic_matrix: dict[int, list[float]],
    topic_labels: list[str],
    output_path: Optional[Path] = None,
) -> str:
    """Export topic evolution over years as TikZ stacked area chart.

    Args:
        year_topic_matrix: Dictionary mapping years to topic proportions.
        topic_labels: List of topic labels.
        output_path: Path to save the TikZ file.

    Returns:
        TikZ code as string.
    """
    years = sorted(year_topic_matrix.keys())
    n_topics = len(topic_labels)

    # Generate plot data for each topic
    plot_commands = []
    colors = ["ldacolor", "nmfcolor", "bertopiccolor", "processcolor", "datacolor"]

    for i, label in enumerate(topic_labels[:5]):  # Limit to 5 topics for readability
        color = colors[i % len(colors)]
        coords = " ".join([
            f"({year}, {year_topic_matrix[year][i]:.4f})"
            for year in years
        ])
        plot_commands.append(
            f"\\addplot[{color}!60, fill={color}!30, thick] coordinates {{{coords}}} "
            f"\\closedcycle;\n\\addlegendentry{{{label}}}"
        )

    tikz_code = r"""% Topic evolution over time
\begin{tikzpicture}
\begin{axis}[
    width=14cm,
    height=8cm,
    xlabel={Year},
    ylabel={Topic Proportion},
    legend pos=outer north east,
    stack plots=y,
    area style,
    xmin=""" + str(min(years)) + r""",
    xmax=""" + str(max(years)) + r""",
    ymin=0,
]
""" + "\n".join(plot_commands) + r"""
\end{axis}
\end{tikzpicture}
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(tikz_code)

    return tikz_code
