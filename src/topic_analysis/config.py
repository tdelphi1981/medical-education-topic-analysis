"""Configuration settings for topic analysis experiments."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LATEX_OUTPUT_DIR = OUTPUT_DIR / "latex"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Data files
CORPUS_FILE = DATA_DIR / "medical_education.csv"

# Ensure output directories exist
for dir_path in [OUTPUT_DIR, LATEX_OUTPUT_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
TOPIC_COUNTS = [5, 10, 15, 20, 25]
RANDOM_SEED = 42

# LDA settings
LDA_CONFIG = {
    "iterations": 1000,
    "alpha": "auto",  # Will be set to 50/K if 'auto'
    "beta": 0.01,
    "passes": 20,
    "eval_every": 10,
}

# NMF settings (scikit-learn)
NMF_CONFIG = {
    "init": "nndsvda",  # NNDSVD with zeros filled with average
    "solver": "cd",  # Coordinate descent
    "beta_loss": "frobenius",  # Frobenius norm
    "max_iter": 200,
    "tol": 1e-4,
    "alpha_W": 0.0,  # L1/L2 regularization for W
    "alpha_H": 0.0,  # L1/L2 regularization for H
    "l1_ratio": 0.0,  # 0=L2, 1=L1
}

# BERTopic settings
BERTOPIC_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.0,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 10,
}

# Preprocessing settings
PREPROCESSING_CONFIG = {
    "min_doc_freq": 5,  # Minimum document frequency
    "max_doc_freq_ratio": 0.5,  # Maximum document frequency ratio
    "max_vocab_size": 10000,
    "min_word_length": 3,
    "spacy_model": "en_core_web_sm",
}

# Evaluation settings
EVALUATION_CONFIG = {
    "coherence_measures": ["c_v", "c_npmi", "u_mass"],
    "top_n_words": 10,  # For coherence and diversity calculations
}

# Column names in the CSV
CSV_COLUMNS = {
    "id": "pmid",
    "title": "title",
    "abstract": "abstract",
    "year": "year",
    "journal": "journal",
    "authors": "authors",
}
