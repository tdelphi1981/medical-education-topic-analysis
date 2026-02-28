#!/usr/bin/env python3
"""Export results as LaTeX tables and TikZ figures.

This script generates all LaTeX content from the experiment results.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_analysis.visualization.tables import (
    generate_topic_table,
    generate_results_table,
    generate_diversity_table,
)
from topic_analysis.visualization.tikz_export import (
    export_coherence_plot,
)
from topic_analysis.config import RESULTS_DIR, LATEX_OUTPUT_DIR, TOPIC_COUNTS


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    if not results_path.exists():
        return {}
    with open(results_path, "r") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Exporting Results to LaTeX")
    print("=" * 60)

    # Load all results
    lda_results = load_results(RESULTS_DIR / "lda_results.json")
    nmf_results = load_results(RESULTS_DIR / "nmf_results.json")
    bertopic_results = load_results(RESULTS_DIR / "bertopic_results.json")

    LATEX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate topic tables for K=10
    print("\nGenerating topic tables (K=10)...")

    if "10" in lda_results:
        table = generate_topic_table(
            lda_results["10"]["topics"],
            "LDA",
            num_words=10,
            output_path=LATEX_OUTPUT_DIR / "table_lda_topics.tex"
        )
        print(f"  LDA topics table saved")

    if "10" in nmf_results:
        table = generate_topic_table(
            nmf_results["10"]["topics"],
            "NMF",
            num_words=10,
            output_path=LATEX_OUTPUT_DIR / "table_nmf_topics.tex"
        )
        print(f"  NMF topics table saved")

    if "10" in bertopic_results:
        table = generate_topic_table(
            bertopic_results["10"]["topics"],
            "BERTopic",
            num_words=10,
            output_path=LATEX_OUTPUT_DIR / "table_bertopic_topics.tex"
        )
        print(f"  BERTopic topics table saved")

    # 2. Generate coherence comparison tables
    print("\nGenerating coherence comparison tables...")

    # Restructure results for table generation
    restructured = {}
    for model_name, results in [("LDA", lda_results), ("NMF", nmf_results), ("BERTopic", bertopic_results)]:
        restructured[model_name] = {}
        for k in TOPIC_COUNTS:
            str_k = str(k)
            if str_k in results:
                restructured[model_name][k] = results[str_k]

    for metric in ["c_v", "c_npmi", "u_mass"]:
        table = generate_results_table(
            restructured,
            TOPIC_COUNTS,
            metric=metric,
            output_path=LATEX_OUTPUT_DIR / f"table_{metric}_coherence.tex"
        )
        print(f"  {metric} coherence table saved")

    # 3. Generate diversity table
    print("\nGenerating diversity table...")
    table = generate_diversity_table(
        restructured,
        TOPIC_COUNTS,
        output_path=LATEX_OUTPUT_DIR / "table_diversity.tex"
    )
    print(f"  Diversity table saved")

    # 4. Generate TikZ coherence plot
    print("\nGenerating TikZ coherence plot...")

    # Prepare data for plotting
    plot_data = {}
    for model_name, results in restructured.items():
        plot_data[model_name] = {}
        for k, metrics in results.items():
            if "coherence" in metrics and metrics["coherence"]:
                cv = metrics["coherence"].get("c_v")
                if cv is not None:
                    plot_data[model_name][k] = cv

    if any(plot_data.values()):
        tikz = export_coherence_plot(
            plot_data,
            TOPIC_COUNTS,
            metric="c_v",
            output_path=LATEX_OUTPUT_DIR / "figure_coherence_comparison.tex"
        )
        print(f"  Coherence plot saved")

    # 5. Print summary of generated files
    print("\n" + "=" * 40)
    print("Generated Files:")
    print("=" * 40)

    for f in sorted(LATEX_OUTPUT_DIR.glob("*.tex")):
        print(f"  {f.name}")

    for f in sorted(LATEX_OUTPUT_DIR.glob("*.json")):
        print(f"  {f.name}")

    print(f"\nOutput directory: {LATEX_OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
