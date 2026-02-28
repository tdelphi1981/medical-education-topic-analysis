"""NMF model training using scikit-learn.

This module provides NMF topic modeling using sklearn's NMF class,
with the same preprocessed data as LDA for objective comparison.
"""

from pathlib import Path
from typing import Optional
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from gensim.corpora import Dictionary

from ..config import NMF_CONFIG, RANDOM_SEED


def corpus_to_sparse_matrix(corpus: list, num_terms: int) -> csr_matrix:
    """Convert gensim corpus to scipy sparse matrix.

    Args:
        corpus: Gensim corpus (list of bag-of-words).
        num_terms: Vocabulary size.

    Returns:
        Sparse document-term matrix (documents x terms).
    """
    data = []
    row_indices = []
    col_indices = []

    for doc_idx, doc in enumerate(corpus):
        for term_id, count in doc:
            row_indices.append(doc_idx)
            col_indices.append(term_id)
            data.append(count)

    return csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(corpus), num_terms),
        dtype=np.float64
    )


def train_nmf(
    corpus: list,
    dictionary: Dictionary,
    num_topics: int,
    init: str = None,
    solver: str = None,
    beta_loss: str = None,
    max_iter: int = None,
    tol: float = None,
    alpha_W: float = None,
    alpha_H: float = None,
    l1_ratio: float = None,
    save_path: Optional[Path] = None,
) -> tuple[NMF, csr_matrix, Dictionary]:
    """Train an NMF model using scikit-learn.

    Uses the same preprocessed data as LDA (gensim Dictionary and corpus)
    to ensure objective comparison between methods.

    Args:
        corpus: Gensim corpus (list of bag-of-words).
        dictionary: Gensim dictionary.
        num_topics: Number of topics to learn.
        init: Initialization method ('nndsvd', 'nndsvda', 'nndsvdar', 'random').
        solver: Optimization solver ('cd' or 'mu').
        beta_loss: Beta divergence loss ('frobenius', 'kullback-leibler', 'itakura-saito').
        max_iter: Maximum number of iterations.
        tol: Tolerance for stopping condition.
        alpha_W: Regularization parameter for W matrix.
        alpha_H: Regularization parameter for H matrix.
        l1_ratio: Ratio of L1 regularization (0=L2 only, 1=L1 only).
        save_path: Path to save the trained model.

    Returns:
        Tuple of (trained NMF model, document-term matrix, dictionary).
    """
    # Set defaults from config
    init = init or NMF_CONFIG.get("init", "nndsvda")
    solver = solver or NMF_CONFIG.get("solver", "cd")
    beta_loss = beta_loss or NMF_CONFIG.get("beta_loss", "frobenius")
    max_iter = max_iter or NMF_CONFIG.get("max_iter", 200)
    tol = tol if tol is not None else NMF_CONFIG.get("tol", 1e-4)
    alpha_W = alpha_W if alpha_W is not None else NMF_CONFIG.get("alpha_W", 0.0)
    alpha_H = alpha_H if alpha_H is not None else NMF_CONFIG.get("alpha_H", 0.0)
    l1_ratio = l1_ratio if l1_ratio is not None else NMF_CONFIG.get("l1_ratio", 0.0)

    print(f"Training NMF (sklearn) with {num_topics} topics...")
    print(f"  Init: {init}, Solver: {solver}, Beta loss: {beta_loss}")
    print(f"  Max iterations: {max_iter}, Tolerance: {tol}")

    # Convert gensim corpus to sparse matrix
    dtm = corpus_to_sparse_matrix(corpus, len(dictionary))
    print(f"  Document-term matrix: {dtm.shape[0]} docs x {dtm.shape[1]} terms")

    # Train NMF model
    model = NMF(
        n_components=num_topics,
        init=init,
        solver=solver,
        beta_loss=beta_loss,
        max_iter=max_iter,
        tol=tol,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        random_state=RANDOM_SEED,
    )

    # Fit the model
    W = model.fit_transform(dtm)

    print(f"  Reconstruction error: {model.reconstruction_err_:.4f}")
    print(f"  Iterations: {model.n_iter_}")

    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "model": model,
                "W": W,
                "dictionary": dictionary,
            }, f)
        print(f"Model saved to {save_path}")

    return model, dtm, dictionary


def get_nmf_topics(
    model: NMF,
    dictionary: Dictionary,
    num_words: int = 10,
) -> list:
    """Get topics from a trained NMF model.

    Args:
        model: Trained sklearn NMF model.
        dictionary: Gensim dictionary.
        num_words: Number of top words per topic.

    Returns:
        List of topics with words and formatted strings.
    """
    topics = []

    # H matrix: topic-term weights (num_topics x num_terms)
    H = model.components_

    for topic_id in range(H.shape[0]):
        # Get top word indices for this topic
        top_indices = H[topic_id].argsort()[::-1][:num_words]
        words = [dictionary[idx] for idx in top_indices if idx in dictionary]

        topics.append({
            "id": topic_id,
            "words": words,
            "formatted": ", ".join(words),
        })

    return topics


def get_document_topics(
    model: NMF,
    dtm: csr_matrix,
    normalize: bool = True,
) -> np.ndarray:
    """Get topic distributions for all documents.

    Args:
        model: Trained NMF model.
        dtm: Document-term matrix.
        normalize: Whether to normalize distributions to sum to 1.

    Returns:
        Document-topic matrix (num_docs x num_topics).
    """
    W = model.transform(dtm)

    if normalize:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

    return W
