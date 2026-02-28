"""Topic diversity evaluation metrics."""

from ..config import EVALUATION_CONFIG


def compute_topic_diversity(
    topics: list[list[str]],
    top_n: int = None,
) -> float:
    """Compute topic diversity as the proportion of unique words.

    Topic diversity measures how distinct topics are from each other.
    A score of 1.0 means all words are unique across topics.
    A score close to 0 means topics share many words.

    Args:
        topics: List of topics, where each topic is a list of words.
        top_n: Number of top words to consider per topic.

    Returns:
        Diversity score between 0 and 1.
    """
    top_n = top_n or EVALUATION_CONFIG["top_n_words"]

    # Get top-n words from each topic
    all_words = []
    for topic in topics:
        all_words.extend(topic[:top_n])

    if not all_words:
        return 0.0

    # Compute proportion of unique words
    unique_words = set(all_words)
    diversity = len(unique_words) / len(all_words)

    return diversity


def compute_pairwise_topic_similarity(
    topics: list[list[str]],
    top_n: int = None,
) -> list[list[float]]:
    """Compute pairwise Jaccard similarity between topics.

    Args:
        topics: List of topics, where each topic is a list of words.
        top_n: Number of top words to consider per topic.

    Returns:
        Similarity matrix as nested lists.
    """
    top_n = top_n or EVALUATION_CONFIG["top_n_words"]

    n_topics = len(topics)
    similarity_matrix = [[0.0] * n_topics for _ in range(n_topics)]

    topic_sets = [set(topic[:top_n]) for topic in topics]

    for i in range(n_topics):
        for j in range(n_topics):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                intersection = len(topic_sets[i] & topic_sets[j])
                union = len(topic_sets[i] | topic_sets[j])
                similarity_matrix[i][j] = intersection / union if union > 0 else 0.0

    return similarity_matrix


def compute_redundancy(
    topics: list[list[str]],
    top_n: int = None,
) -> float:
    """Compute topic redundancy (complement of diversity).

    Args:
        topics: List of topics.
        top_n: Number of top words per topic.

    Returns:
        Redundancy score between 0 and 1.
    """
    return 1.0 - compute_topic_diversity(topics, top_n)
