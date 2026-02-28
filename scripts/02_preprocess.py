#!/usr/bin/env python3
"""Preprocess corpus for topic modeling.

This script tokenizes, cleans, and prepares the corpus for LDA and NMF.
"""

import sys
import pickle
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_analysis.data.loader import get_text_for_topic_modeling
from topic_analysis.data.preprocessing import (
    preprocess_corpus,
    add_bigrams,
    create_dictionary_and_corpus,
    get_preprocessing_stats,
)
from topic_analysis.config import CORPUS_FILE, MODELS_DIR


def main():
    print("=" * 60)
    print("Preprocessing Medical Education Corpus")
    print("=" * 60)

    # Check if data file exists
    if not CORPUS_FILE.exists():
        print(f"Error: Data file not found: {CORPUS_FILE}")
        sys.exit(1)

    # Load raw texts
    print("\nLoading documents...")
    raw_texts = get_text_for_topic_modeling(include_title=False)
    print(f"  Loaded {len(raw_texts):,} documents")

    # Preprocess
    print("\nPreprocessing (tokenization, lemmatization, stopword removal)...")
    preprocessed = preprocess_corpus(raw_texts, show_progress=True)
    print(f"  Preprocessed {len(preprocessed):,} non-empty documents")

    # Add bigrams
    print("\nDetecting and adding bigrams...")
    preprocessed_bigrams = add_bigrams(preprocessed)

    # Create dictionary and corpus
    print("\nCreating dictionary and corpus...")
    dictionary, corpus = create_dictionary_and_corpus(
        preprocessed_bigrams,
        save_path=MODELS_DIR / "dictionary.gensim"
    )

    # Get stats
    stats = get_preprocessing_stats(raw_texts, preprocessed_bigrams, dictionary)

    print(f"\nPreprocessing Statistics:")
    print(f"  Raw documents: {stats['raw_documents']:,}")
    print(f"  Preprocessed documents: {stats['preprocessed_documents']:,}")
    print(f"  Vocabulary size: {stats['vocabulary_size']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Average tokens per document: {stats['avg_tokens_per_doc']:.1f}")

    # Save preprocessed data
    output_path = MODELS_DIR / "preprocessed_data.pkl"
    print(f"\nSaving preprocessed data to: {output_path}")

    with open(output_path, "wb") as f:
        pickle.dump({
            "texts": preprocessed_bigrams,
            "corpus": corpus,
            "dictionary_path": str(MODELS_DIR / "dictionary.gensim"),
            "stats": stats,
        }, f)

    # Also save texts for BERTopic (needs raw texts)
    raw_texts_path = MODELS_DIR / "raw_texts.pkl"
    print(f"Saving raw texts for BERTopic to: {raw_texts_path}")

    with open(raw_texts_path, "wb") as f:
        pickle.dump(raw_texts, f)

    print("\nDone!")


if __name__ == "__main__":
    main()
