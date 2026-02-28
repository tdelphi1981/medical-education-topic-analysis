"""Text preprocessing utilities for topic modeling."""

import re
from typing import Optional
from pathlib import Path

import spacy
from gensim.corpora import Dictionary
from gensim.models import Phrases
from tqdm import tqdm

from ..config import PREPROCESSING_CONFIG, MODELS_DIR


# Lazy loading of spaCy model
_nlp = None


def get_nlp():
    """Get or initialize the spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(PREPROCESSING_CONFIG["spacy_model"])
        except OSError:
            print(f"Downloading spaCy model: {PREPROCESSING_CONFIG['spacy_model']}")
            from spacy.cli import download
            download(PREPROCESSING_CONFIG["spacy_model"])
            _nlp = spacy.load(PREPROCESSING_CONFIG["spacy_model"])
    return _nlp


def clean_text(text: str) -> str:
    """Basic text cleaning.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters but keep spaces and letters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(
    text: str,
    min_length: int = None,
    remove_stopwords: bool = True
) -> list[str]:
    """Tokenize and lemmatize text using spaCy.

    Args:
        text: Cleaned text string.
        min_length: Minimum token length to keep.
        remove_stopwords: Whether to remove stopwords.

    Returns:
        List of lemmatized tokens.
    """
    min_length = min_length or PREPROCESSING_CONFIG["min_word_length"]
    nlp = get_nlp()

    doc = nlp(text)

    tokens = []
    for token in doc:
        # Skip if stopword (optional)
        if remove_stopwords and token.is_stop:
            continue

        # Skip punctuation and spaces
        if token.is_punct or token.is_space:
            continue

        # Skip short tokens
        if len(token.lemma_) < min_length:
            continue

        # Skip tokens that are purely numeric
        if token.lemma_.isdigit():
            continue

        tokens.append(token.lemma_)

    return tokens


def preprocess_corpus(
    documents: list[str],
    show_progress: bool = True
) -> list[list[str]]:
    """Preprocess a corpus of documents.

    Args:
        documents: List of raw text documents.
        show_progress: Whether to show progress bar.

    Returns:
        List of tokenized documents.
    """
    preprocessed = []

    iterator = tqdm(documents, desc="Preprocessing") if show_progress else documents

    for doc in iterator:
        cleaned = clean_text(doc)
        tokens = tokenize_and_lemmatize(cleaned)
        if tokens:  # Only add non-empty documents
            preprocessed.append(tokens)

    return preprocessed


def add_bigrams(
    documents: list[list[str]],
    min_count: int = 5,
    threshold: float = 10.0
) -> list[list[str]]:
    """Add common bigrams to tokenized documents.

    Args:
        documents: List of tokenized documents.
        min_count: Minimum bigram frequency.
        threshold: Bigram detection threshold.

    Returns:
        Documents with bigrams added.
    """
    # Train bigram model
    bigram = Phrases(documents, min_count=min_count, threshold=threshold)
    bigram_mod = bigram.freeze()

    return [bigram_mod[doc] for doc in documents]


def create_dictionary_and_corpus(
    documents: list[list[str]],
    no_below: int = None,
    no_above: float = None,
    keep_n: int = None,
    save_path: Optional[Path] = None
) -> tuple[Dictionary, list]:
    """Create Gensim dictionary and corpus from preprocessed documents.

    Args:
        documents: List of tokenized documents.
        no_below: Minimum document frequency.
        no_above: Maximum document frequency ratio.
        keep_n: Maximum vocabulary size.
        save_path: Path to save dictionary.

    Returns:
        Tuple of (dictionary, corpus).
    """
    no_below = no_below or PREPROCESSING_CONFIG["min_doc_freq"]
    no_above = no_above or PREPROCESSING_CONFIG["max_doc_freq_ratio"]
    keep_n = keep_n or PREPROCESSING_CONFIG["max_vocab_size"]

    # Create dictionary
    dictionary = Dictionary(documents)

    # Filter extremes
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

    # Create bag-of-words corpus
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Save dictionary if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dictionary.save(str(save_path))

    return dictionary, corpus


def get_preprocessing_stats(
    raw_documents: list[str],
    preprocessed_documents: list[list[str]],
    dictionary: Dictionary
) -> dict:
    """Get statistics about preprocessing.

    Args:
        raw_documents: Original documents.
        preprocessed_documents: Preprocessed tokenized documents.
        dictionary: Gensim dictionary.

    Returns:
        Dictionary with preprocessing statistics.
    """
    total_tokens = sum(len(doc) for doc in preprocessed_documents)
    avg_tokens = total_tokens / len(preprocessed_documents) if preprocessed_documents else 0

    return {
        "raw_documents": len(raw_documents),
        "preprocessed_documents": len(preprocessed_documents),
        "vocabulary_size": len(dictionary),
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": avg_tokens,
    }
