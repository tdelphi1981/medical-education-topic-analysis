"""Statistics extraction and export utilities."""

import json
from pathlib import Path
from typing import Optional

from .loader import get_corpus_stats
from ..config import CORPUS_FILE, LATEX_OUTPUT_DIR


def export_stats_to_json(
    stats: Optional[dict] = None,
    filepath: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> dict:
    """Export corpus statistics to JSON file.

    Args:
        stats: Pre-computed statistics. If None, computed from data.
        filepath: Path to the CSV file.
        output_path: Path for output JSON file.

    Returns:
        Dictionary with statistics.
    """
    if stats is None:
        stats = get_corpus_stats(filepath)

    output_path = output_path or LATEX_OUTPUT_DIR / "corpus_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    return stats


def generate_latex_stats_table(stats: dict) -> str:
    """Generate LaTeX table with corpus statistics.

    Args:
        stats: Dictionary with corpus statistics.

    Returns:
        LaTeX table code as string.
    """
    year_min, year_max = stats["year_range"]

    table = r"""
\begin{table}[htbp]
\centering
\caption{Summary statistics of the medical education corpus}
\label{tab:dataset-stats}
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Total records & """ + f"{stats['total_records']:,}" + r""" \\
Records with abstracts & """ + f"{stats['with_abstract']:,}" + r""" \\
Year range & """ + f"{year_min}--{year_max}" + r""" \\
Unique journals & """ + f"{stats['unique_journals']:,}" + r""" \\
Mean abstract length (chars) & """ + f"{stats['abstract_char_length']['mean']:.1f}" + r""" \\
Median abstract length (chars) & """ + f"{stats['abstract_char_length']['median']:.1f}" + r""" \\
Mean word count (approx.) & """ + f"{stats['abstract_word_count']['mean']:.1f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def generate_latex_journal_table(stats: dict, top_n: int = 10) -> str:
    """Generate LaTeX table with top journals.

    Args:
        stats: Dictionary with corpus statistics.
        top_n: Number of top journals to include.

    Returns:
        LaTeX table code as string.
    """
    journals = stats["top_journals"][:top_n]

    rows = []
    for j in journals:
        journal_name = j["journal"][:50] + "..." if len(j["journal"]) > 50 else j["journal"]
        rows.append(f"{journal_name} & {j['count']:,} \\\\")

    table = r"""
\begin{table}[htbp]
\centering
\caption{Top """ + str(top_n) + r""" journals by publication count}
\label{tab:top-journals}
\begin{tabular}{lr}
\toprule
\textbf{Journal} & \textbf{Count} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def export_latex_tables(
    stats: Optional[dict] = None,
    output_dir: Optional[Path] = None
) -> None:
    """Export all LaTeX tables for the chapter.

    Args:
        stats: Dictionary with corpus statistics. If None, computed from data.
        output_dir: Directory for output files.
    """
    if stats is None:
        stats = get_corpus_stats()

    output_dir = output_dir or LATEX_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset stats table
    with open(output_dir / "table_dataset_stats.tex", "w") as f:
        f.write(generate_latex_stats_table(stats))

    # Top journals table
    with open(output_dir / "table_top_journals.tex", "w") as f:
        f.write(generate_latex_journal_table(stats))

    print(f"LaTeX tables exported to {output_dir}")
