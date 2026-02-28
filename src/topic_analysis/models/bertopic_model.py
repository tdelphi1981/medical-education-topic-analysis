"""BERTopic model training."""

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import BERTOPIC_CONFIG, MODELS_DIR, RANDOM_SEED


def train_bertopic(
    documents: list[str],
    min_topic_size: int = None,
    nr_topics: Optional[int] = None,
    embedding_model: str = None,
    umap_n_neighbors: int = None,
    umap_n_components: int = None,
    umap_min_dist: float = None,
    hdbscan_min_cluster_size: int = None,
    hdbscan_min_samples: int = None,
    save_path: Optional[Path] = None,
):
    """Train a BERTopic model.

    Args:
        documents: List of raw text documents.
        min_topic_size: Minimum size for a topic.
        nr_topics: Target number of topics (None for automatic).
        embedding_model: Sentence transformer model name.
        umap_n_neighbors: UMAP n_neighbors parameter.
        umap_n_components: UMAP n_components parameter.
        umap_min_dist: UMAP min_dist parameter.
        hdbscan_min_cluster_size: HDBSCAN min_cluster_size.
        hdbscan_min_samples: HDBSCAN min_samples.
        save_path: Path to save the trained model.

    Returns:
        Tuple of (BERTopic model, topics, probabilities).
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
    except ImportError as e:
        raise ImportError(
            f"Required package not installed: {e}. "
            "Install with: pip install bertopic sentence-transformers umap-learn hdbscan"
        )

    # Set defaults from config
    embedding_model = embedding_model or BERTOPIC_CONFIG["embedding_model"]
    umap_n_neighbors = umap_n_neighbors or BERTOPIC_CONFIG["umap_n_neighbors"]
    umap_n_components = umap_n_components or BERTOPIC_CONFIG["umap_n_components"]
    umap_min_dist = umap_min_dist if umap_min_dist is not None else BERTOPIC_CONFIG["umap_min_dist"]
    hdbscan_min_cluster_size = hdbscan_min_cluster_size or BERTOPIC_CONFIG["hdbscan_min_cluster_size"]
    hdbscan_min_samples = hdbscan_min_samples or BERTOPIC_CONFIG["hdbscan_min_samples"]

    print(f"Training BERTopic...")
    print(f"  Embedding model: {embedding_model}")
    print(f"  UMAP: n_neighbors={umap_n_neighbors}, n_components={umap_n_components}")
    print(f"  HDBSCAN: min_cluster_size={hdbscan_min_cluster_size}")

    # Initialize components
    sentence_model = SentenceTransformer(embedding_model)

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        top_n_words=10,
        verbose=True,
    )

    # Fit the model
    topics, probs = topic_model.fit_transform(documents)

    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        topic_model.save(str(save_path))
        print(f"Model saved to {save_path}")

    return topic_model, topics, probs


def get_bertopic_topics(
    topic_model,
    num_words: int = 10
) -> list:
    """Extract formatted topics from BERTopic model.

    Args:
        topic_model: Trained BERTopic model.
        num_words: Number of top words per topic.

    Returns:
        List of topics with words and formatted strings.
    """
    topics = []
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:  # Skip outlier topic
            continue

        topic_words = topic_model.get_topic(topic_id)
        words = [word for word, _ in topic_words[:num_words]]

        topics.append({
            "id": topic_id,
            "words": words,
            "formatted": ", ".join(words),
            "count": row["Count"],
        })

    return topics


def get_bertopic_document_topics(
    topic_model,
    documents: list[str],
) -> tuple[list, np.ndarray]:
    """Get topic assignments for documents.

    Args:
        topic_model: Trained BERTopic model.
        documents: List of documents.

    Returns:
        Tuple of (topic assignments, probabilities).
    """
    topics, probs = topic_model.transform(documents)
    return topics, probs
