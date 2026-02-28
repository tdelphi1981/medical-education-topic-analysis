#!/usr/bin/env python3
"""Extract and export dataset statistics.

This script analyzes the medical education corpus and generates
statistics and LaTeX tables for the chapter.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_analysis.data.loader import get_corpus_stats
from topic_analysis.data.stats import (
    export_stats_to_json,
    export_latex_tables,
)
from topic_analysis.config import CORPUS_FILE, LATEX_OUTPUT_DIR


def main():
    print("=" * 60)
    print("Medical Education Corpus Statistics")
    print("=" * 60)

    # Check if data file exists
    if not CORPUS_FILE.exists():
        print(f"Error: Data file not found: {CORPUS_FILE}")
        sys.exit(1)

    print(f"\nLoading data from: {CORPUS_FILE}")

    # Get statistics
    stats = get_corpus_stats()

    # Print summary
    print(f"\nBasic Statistics:")
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Records with abstracts: {stats['with_abstract']:,}")
    print(f"  Unique journals: {stats['unique_journals']:,}")

    year_min, year_max = stats['year_range']
    print(f"  Year range: {year_min} - {year_max}")

    print(f"\nAbstract Statistics:")
    char_stats = stats['abstract_char_length']
    print(f"  Mean character length: {char_stats['mean']:.1f}")
    print(f"  Median character length: {char_stats['median']:.1f}")
    print(f"  Std dev: {char_stats['std']:.1f}")

    word_stats = stats['abstract_word_count']
    print(f"  Mean word count (approx): {word_stats['mean']:.1f}")
    print(f"  Median word count (approx): {word_stats['median']:.1f}")

    print(f"\nTop 10 Journals:")
    for i, journal in enumerate(stats['top_journals'][:10], 1):
        name = journal['journal'][:50]
        if len(journal['journal']) > 50:
            name += "..."
        print(f"  {i}. {name}: {journal['count']:,}")

    # Export to JSON
    print(f"\nExporting statistics to JSON...")
    export_stats_to_json(stats=stats)

    # Export LaTeX tables
    print(f"Generating LaTeX tables...")
    export_latex_tables(stats=stats)

    print(f"\nOutput saved to: {LATEX_OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
