"""LDA model training and inference using Gensim."""

from pathlib import Path
from typing import Optional

from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, LdaModel
import numpy as np

from ..config import LDA_CONFIG, MODELS_DIR, RANDOM_SEED


def train_lda(
    corpus: list,
    dictionary: Dictionary,
    num_topics: int,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    iterations: int = None,
    passes: int = None,
    workers: int = None,
    save_path: Optional[Path] = None,
    eval_every: int = None,
) -> LdaMulticore:
    """Train an LDA model using Gensim.

    Args:
        corpus: Gensim corpus (list of bag-of-words).
        dictionary: Gensim dictionary.
        num_topics: Number of topics to learn.
        alpha: Document-topic Dirichlet prior. If None, uses 50/num_topics.
        beta: Topic-word Dirichlet prior. If None, uses config default.
        iterations: Number of iterations per document.
        passes: Number of passes through the corpus.
        workers: Number of worker processes.
        save_path: Path to save the trained model.
        eval_every: Evaluate perplexity every N documents.

    Returns:
        Trained LDA model.
    """
    # Set defaults
    if alpha is None:
        alpha = 50.0 / num_topics if LDA_CONFIG["alpha"] == "auto" else LDA_CONFIG["alpha"]
    if beta is None:
        beta = LDA_CONFIG["beta"]
    if iterations is None:
        iterations = LDA_CONFIG["iterations"]
    if passes is None:
        passes = LDA_CONFIG["passes"]
    if eval_every is None:
        eval_every = LDA_CONFIG["eval_every"]

    print(f"Training LDA with {num_topics} topics...")
    print(f"  Alpha: {alpha}, Beta: {beta}")
    print(f"  Iterations: {iterations}, Passes: {passes}")

    # Use multicore if workers > 1
    if workers and workers > 1:
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            alpha=alpha,
            eta=beta,
            iterations=iterations,
            passes=passes,
            workers=workers,
            random_state=RANDOM_SEED,
            eval_every=eval_every,
        )
    else:
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            alpha=alpha,
            eta=beta,
            iterations=iterations,
            passes=passes,
            random_state=RANDOM_SEED,
            eval_every=eval_every,
        )

    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"Model saved to {save_path}")

    return model


def get_lda_topics(
    model: LdaModel,
    num_words: int = 10,
    formatted: bool = True
) -> list:
    """Get topics from a trained LDA model.

    Args:
        model: Trained LDA model.
        num_words: Number of top words per topic.
        formatted: If True, return formatted strings; otherwise, (word, prob) tuples.

    Returns:
        List of topics.
    """
    topics = []

    for topic_id in range(model.num_topics):
        if formatted:
            topic_words = model.show_topic(topic_id, topn=num_words)
            words = [word for word, _ in topic_words]
            topics.append({
                "id": topic_id,
                "words": words,
                "formatted": ", ".join(words)
            })
        else:
            topic_words = model.show_topic(topic_id, topn=num_words)
            topics.append({
                "id": topic_id,
                "words": topic_words
            })

    return topics


def get_document_topics(
    model: LdaModel,
    corpus: list,
    minimum_probability: float = 0.01
) -> list:
    """Get topic distributions for all documents.

    Args:
        model: Trained LDA model.
        corpus: Gensim corpus.
        minimum_probability: Minimum probability threshold.

    Returns:
        List of topic distributions per document.
    """
    doc_topics = []

    for doc in corpus:
        topic_dist = model.get_document_topics(doc, minimum_probability=minimum_probability)
        # Convert to full distribution array
        full_dist = np.zeros(model.num_topics)
        for topic_id, prob in topic_dist:
            full_dist[topic_id] = prob
        doc_topics.append(full_dist)

    return doc_topics


def compute_perplexity(model: LdaModel, corpus: list) -> float:
    """Compute perplexity of the model on a corpus.

    Args:
        model: Trained LDA model.
        corpus: Gensim corpus.

    Returns:
        Perplexity value.
    """
    return model.log_perplexity(corpus)
