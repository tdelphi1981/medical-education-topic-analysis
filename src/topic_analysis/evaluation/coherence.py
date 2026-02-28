"""Topic coherence evaluation metrics."""

from typing import Optional

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from ..config import EVALUATION_CONFIG


def compute_coherence(
    topics: list[list[str]],
    texts: list[list[str]],
    dictionary: Dictionary,
    coherence_type: str = "c_v",
) -> float:
    """Compute coherence score for a set of topics.

    Args:
        topics: List of topics, where each topic is a list of words.
        texts: Tokenized corpus (list of token lists).
        dictionary: Gensim dictionary.
        coherence_type: Type of coherence measure ("c_v", "c_npmi", "u_mass").

    Returns:
        Coherence score.
    """
    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type,
    )

    return coherence_model.get_coherence()


def compute_all_coherence_metrics(
    topics: list[list[str]],
    texts: list[list[str]],
    dictionary: Dictionary,
    measures: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute multiple coherence metrics for topics.

    Args:
        topics: List of topics, where each topic is a list of words.
        texts: Tokenized corpus (list of token lists).
        dictionary: Gensim dictionary.
        measures: List of coherence measures to compute.

    Returns:
        Dictionary mapping measure names to scores.
    """
    measures = measures or EVALUATION_CONFIG["coherence_measures"]
    results = {}

    for measure in measures:
        try:
            score = compute_coherence(topics, texts, dictionary, measure)
            results[measure] = score
        except Exception as e:
            print(f"Warning: Could not compute {measure} coherence: {e}")
            results[measure] = None

    return results


def compute_coherence_per_topic(
    topics: list[list[str]],
    texts: list[list[str]],
    dictionary: Dictionary,
    coherence_type: str = "c_v",
) -> list[float]:
    """Compute coherence score for each individual topic.

    Args:
        topics: List of topics, where each topic is a list of words.
        texts: Tokenized corpus (list of token lists).
        dictionary: Gensim dictionary.
        coherence_type: Type of coherence measure.

    Returns:
        List of coherence scores, one per topic.
    """
    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type,
    )

    return coherence_model.get_coherence_per_topic()


def compute_lda_coherence(
    model,
    corpus: list,
    texts: list[list[str]],
    dictionary: Dictionary,
    coherence_type: str = "c_v",
) -> float:
    """Compute coherence directly from an LDA model.

    Args:
        model: Trained Gensim LDA model.
        corpus: Gensim corpus.
        texts: Tokenized corpus.
        dictionary: Gensim dictionary.
        coherence_type: Type of coherence measure.

    Returns:
        Coherence score.
    """
    coherence_model = CoherenceModel(
        model=model,
        corpus=corpus,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type,
    )

    return coherence_model.get_coherence()


def compute_nmf_coherence(
    model,
    corpus: list,
    texts: list[list[str]],
    dictionary: Dictionary,
    coherence_type: str = "c_v",
) -> float:
    """Compute coherence directly from a Gensim NMF model.

    Args:
        model: Trained Gensim NMF model.
        corpus: Gensim corpus.
        texts: Tokenized corpus.
        dictionary: Gensim dictionary.
        coherence_type: Type of coherence measure.

    Returns:
        Coherence score.
    """
    coherence_model = CoherenceModel(
        model=model,
        corpus=corpus,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type,
    )

    return coherence_model.get_coherence()
