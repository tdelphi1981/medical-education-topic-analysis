"""Data loading utilities using DuckDB."""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional

from ..config import CORPUS_FILE, CSV_COLUMNS


def load_corpus(
    filepath: Optional[Path] = None,
    with_abstract_only: bool = True
) -> pd.DataFrame:
    """Load the medical education corpus using DuckDB.

    Args:
        filepath: Path to the CSV file. Defaults to CORPUS_FILE.
        with_abstract_only: If True, filter to records with non-empty abstracts.

    Returns:
        DataFrame with corpus data.
    """
    filepath = filepath or CORPUS_FILE

    conn = duckdb.connect(":memory:")

    # Load CSV into DuckDB
    query = f"""
    SELECT *
    FROM read_csv_auto('{filepath}', header=true)
    """

    if with_abstract_only:
        query = f"""
        SELECT *
        FROM read_csv_auto('{filepath}', header=true)
        WHERE {CSV_COLUMNS['abstract']} IS NOT NULL
          AND TRIM({CSV_COLUMNS['abstract']}) != ''
        """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def get_corpus_stats(filepath: Optional[Path] = None) -> dict:
    """Get comprehensive statistics about the corpus.

    Args:
        filepath: Path to the CSV file. Defaults to CORPUS_FILE.

    Returns:
        Dictionary with corpus statistics.
    """
    filepath = filepath or CORPUS_FILE

    conn = duckdb.connect(":memory:")

    # Create view for the corpus
    conn.execute(f"""
        CREATE VIEW corpus AS
        SELECT * FROM read_csv_auto('{filepath}', header=true)
    """)

    stats = {}

    # Basic counts
    result = conn.execute("""
        SELECT
            COUNT(*) as total_records,
            COUNT(CASE WHEN abstract IS NOT NULL AND TRIM(abstract) != '' THEN 1 END) as with_abstract,
            COUNT(DISTINCT journal) as unique_journals
        FROM corpus
    """).fetchone()
    stats["total_records"] = result[0]
    stats["with_abstract"] = result[1]
    stats["unique_journals"] = result[2]

    # Year range
    result = conn.execute("""
        SELECT MIN(year) as min_year, MAX(year) as max_year
        FROM corpus
        WHERE year IS NOT NULL
    """).fetchone()
    stats["year_range"] = (result[0], result[1])

    # Year distribution
    year_dist = conn.execute("""
        SELECT year, COUNT(*) as count
        FROM corpus
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year
    """).fetchdf()
    stats["year_distribution"] = year_dist.to_dict("records")

    # Abstract length statistics (character count)
    result = conn.execute("""
        SELECT
            AVG(LENGTH(abstract)) as mean_char_len,
            MEDIAN(LENGTH(abstract)) as median_char_len,
            STDDEV(LENGTH(abstract)) as std_char_len,
            MIN(LENGTH(abstract)) as min_char_len,
            MAX(LENGTH(abstract)) as max_char_len
        FROM corpus
        WHERE abstract IS NOT NULL AND TRIM(abstract) != ''
    """).fetchone()
    stats["abstract_char_length"] = {
        "mean": result[0],
        "median": result[1],
        "std": result[2],
        "min": result[3],
        "max": result[4],
    }

    # Word count statistics (approximate using space splitting)
    result = conn.execute("""
        SELECT
            AVG(LENGTH(abstract) - LENGTH(REPLACE(abstract, ' ', '')) + 1) as mean_word_count,
            MEDIAN(LENGTH(abstract) - LENGTH(REPLACE(abstract, ' ', '')) + 1) as median_word_count
        FROM corpus
        WHERE abstract IS NOT NULL AND TRIM(abstract) != ''
    """).fetchone()
    stats["abstract_word_count"] = {
        "mean": result[0],
        "median": result[1],
    }

    # Top journals
    top_journals = conn.execute("""
        SELECT journal, COUNT(*) as count
        FROM corpus
        WHERE journal IS NOT NULL
        GROUP BY journal
        ORDER BY count DESC
        LIMIT 20
    """).fetchdf()
    stats["top_journals"] = top_journals.to_dict("records")

    # Publication types distribution
    pub_types = conn.execute("""
        SELECT publication_types, COUNT(*) as count
        FROM corpus
        WHERE publication_types IS NOT NULL
        GROUP BY publication_types
        ORDER BY count DESC
        LIMIT 20
    """).fetchdf()
    stats["publication_types"] = pub_types.to_dict("records")

    conn.close()

    return stats


def get_text_for_topic_modeling(
    filepath: Optional[Path] = None,
    text_column: str = "abstract",
    include_title: bool = False
) -> list[str]:
    """Get text documents ready for topic modeling.

    Args:
        filepath: Path to the CSV file. Defaults to CORPUS_FILE.
        text_column: Column to use for text (default: abstract).
        include_title: Whether to prepend title to abstract.

    Returns:
        List of text documents.
    """
    df = load_corpus(filepath, with_abstract_only=True)

    if include_title:
        texts = (df["title"].fillna("") + " " + df[text_column].fillna("")).tolist()
    else:
        texts = df[text_column].fillna("").tolist()

    return texts
