#!/usr/bin/env python3
"""Measure computational performance for all topic modeling methods.

This script measures (with multiple repetitions for reliability):
- Training time (seconds)
- Peak memory usage (GB)
- Inference time (ms per document)
"""

import sys
import pickle
import json
import time
import tracemalloc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np

from topic_analysis.config import MODELS_DIR, RESULTS_DIR, RANDOM_SEED

# Number of repetitions for reliable measurements
NUM_REPETITIONS = 10


def measure_lda_single(corpus, dictionary, num_topics=10, num_inference_docs=100):
    """Single LDA measurement."""
    tracemalloc.start()
    start_time = time.time()

    model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha=50.0 / num_topics,
        eta=0.01,
        iterations=1000,
        passes=20,
        workers=4,
        random_state=RANDOM_SEED,
        eval_every=None,
    )

    training_time = time.time() - start_time
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Inference
    sample_docs = corpus[:num_inference_docs]
    start_time = time.time()
    for doc in sample_docs:
        _ = model.get_document_topics(doc)
    inference_time = (time.time() - start_time) / len(sample_docs) * 1000

    return training_time, peak_memory / (1024 ** 3), inference_time


def measure_nmf_single(dtm_tfidf, num_topics=10, num_inference_docs=100):
    """Single NMF measurement."""
    tracemalloc.start()
    start_time = time.time()

    model = NMF(
        n_components=num_topics,
        init='nndsvda',
        solver='cd',
        beta_loss='frobenius',
        max_iter=200,
        tol=1e-4,
        random_state=RANDOM_SEED,
    )

    W = model.fit_transform(dtm_tfidf)

    training_time = time.time() - start_time
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Inference
    sample_dtm = dtm_tfidf[:num_inference_docs]
    start_time = time.time()
    _ = model.transform(sample_dtm)
    inference_time = (time.time() - start_time) / num_inference_docs * 1000

    return training_time, peak_memory / (1024 ** 3), inference_time


def measure_bertopic_single(documents, embeddings, num_inference_docs=100):
    """Single BERTopic measurement (using pre-computed embeddings)."""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    tracemalloc.start()
    start_time = time.time()

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=RANDOM_SEED,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=10,
        metric='euclidean',
        prediction_data=True,
    )

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=False,
    )

    topics, probs = model.fit_transform(documents, embeddings=embeddings)

    training_time = time.time() - start_time
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Inference (transform only, embeddings pre-computed)
    sample_embeddings = embeddings[:num_inference_docs]
    sample_docs = documents[:num_inference_docs]
    start_time = time.time()
    _ = model.transform(sample_docs, embeddings=sample_embeddings)
    inference_time = (time.time() - start_time) / num_inference_docs * 1000

    return training_time, peak_memory / (1024 ** 3), inference_time


def compute_stats(values):
    """Compute mean and std."""
    return np.mean(values), np.std(values)


def main():
    print("=" * 60)
    print(f"Computational Performance Measurement ({NUM_REPETITIONS} repetitions)")
    print("=" * 60)

    # Load preprocessed data
    preprocessed_path = MODELS_DIR / "preprocessed_data.pkl"
    if not preprocessed_path.exists():
        print(f"Error: Preprocessed data not found. Run 02_preprocess.py first.")
        sys.exit(1)

    print("\nLoading preprocessed data...")
    with open(preprocessed_path, "rb") as f:
        data = pickle.load(f)

    corpus = data["corpus"]
    dictionary = Dictionary.load(data["dictionary_path"])

    # Load raw texts for BERTopic
    raw_texts_path = MODELS_DIR / "raw_texts.pkl"
    with open(raw_texts_path, "rb") as f:
        documents = pickle.load(f)
    documents = [doc for doc in documents if doc and doc.strip()]

    print(f"  Documents: {len(corpus):,}")
    print(f"  Vocabulary: {len(dictionary):,}")

    # Prepare NMF data once
    print("\nPreparing NMF sparse matrix...")
    n_docs = len(corpus)
    n_terms = len(dictionary)
    row_indices, col_indices, data_vals = [], [], []
    for doc_idx, doc in enumerate(corpus):
        for term_id, count in doc:
            row_indices.append(doc_idx)
            col_indices.append(term_id)
            data_vals.append(count)
    dtm = csr_matrix((data_vals, (row_indices, col_indices)), shape=(n_docs, n_terms))
    dtm_tfidf = normalize(dtm, norm='l2')

    # Pre-compute BERTopic embeddings once (this is a fixed cost)
    print("\nPre-computing BERTopic embeddings (one-time cost)...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_start = time.time()
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    embedding_time = time.time() - embed_start
    print(f"  Embedding time: {embedding_time:.2f}s")

    # Storage for all measurements
    lda_times, lda_mems, lda_infers = [], [], []
    nmf_times, nmf_mems, nmf_infers = [], [], []
    bert_times, bert_mems, bert_infers = [], [], []

    # Run measurements
    for i in range(NUM_REPETITIONS):
        print(f"\n{'='*50}")
        print(f"Repetition {i+1}/{NUM_REPETITIONS}")
        print("=" * 50)

        # LDA
        print("  LDA...", end=" ", flush=True)
        t, m, inf = measure_lda_single(corpus, dictionary)
        lda_times.append(t)
        lda_mems.append(m)
        lda_infers.append(inf)
        print(f"done ({t:.2f}s)")

        # NMF
        print("  NMF...", end=" ", flush=True)
        t, m, inf = measure_nmf_single(dtm_tfidf)
        nmf_times.append(t)
        nmf_mems.append(m)
        nmf_infers.append(inf)
        print(f"done ({t:.2f}s)")

        # BERTopic
        print("  BERTopic...", end=" ", flush=True)
        t, m, inf = measure_bertopic_single(documents, embeddings)
        bert_times.append(t)
        bert_mems.append(m)
        bert_infers.append(inf)
        print(f"done ({t:.2f}s)")

    # Compute statistics
    results = {
        "num_repetitions": NUM_REPETITIONS,
        "num_documents": len(corpus),
        "LDA": {
            "training_time_s": {"mean": round(np.mean(lda_times), 2), "std": round(np.std(lda_times), 2)},
            "peak_memory_gb": {"mean": round(np.mean(lda_mems), 3), "std": round(np.std(lda_mems), 3)},
            "inference_time_ms": {"mean": round(np.mean(lda_infers), 3), "std": round(np.std(lda_infers), 3)},
            "raw_values": {"training": lda_times, "memory": lda_mems, "inference": lda_infers}
        },
        "NMF": {
            "training_time_s": {"mean": round(np.mean(nmf_times), 2), "std": round(np.std(nmf_times), 2)},
            "peak_memory_gb": {"mean": round(np.mean(nmf_mems), 3), "std": round(np.std(nmf_mems), 3)},
            "inference_time_ms": {"mean": round(np.mean(nmf_infers), 3), "std": round(np.std(nmf_infers), 3)},
            "raw_values": {"training": nmf_times, "memory": nmf_mems, "inference": nmf_infers}
        },
        "BERTopic": {
            "embedding_time_s": round(embedding_time, 2),
            "training_time_s": {"mean": round(np.mean(bert_times), 2), "std": round(np.std(bert_times), 2)},
            "total_time_s": {"mean": round(embedding_time + np.mean(bert_times), 2), "std": round(np.std(bert_times), 2)},
            "peak_memory_gb": {"mean": round(np.mean(bert_mems), 3), "std": round(np.std(bert_mems), 3)},
            "inference_time_ms": {"mean": round(np.mean(bert_infers), 3), "std": round(np.std(bert_infers), 3)},
            "raw_values": {"training": bert_times, "memory": bert_mems, "inference": bert_infers}
        }
    }

    # Print summary
    print("\n" + "=" * 70)
    print(f"Performance Summary (K=10 topics, {NUM_REPETITIONS} repetitions)")
    print("=" * 70)

    print(f"\n{'Method':<12} {'Training (s)':<20} {'Memory (GB)':<20} {'Inference (ms)':<20}")
    print("-" * 72)

    lda = results["LDA"]
    print(f"{'LDA':<12} {lda['training_time_s']['mean']:.2f} ± {lda['training_time_s']['std']:.2f}       "
          f"{lda['peak_memory_gb']['mean']:.3f} ± {lda['peak_memory_gb']['std']:.3f}       "
          f"{lda['inference_time_ms']['mean']:.3f} ± {lda['inference_time_ms']['std']:.3f}")

    nmf = results["NMF"]
    print(f"{'NMF':<12} {nmf['training_time_s']['mean']:.2f} ± {nmf['training_time_s']['std']:.2f}        "
          f"{nmf['peak_memory_gb']['mean']:.3f} ± {nmf['peak_memory_gb']['std']:.3f}       "
          f"{nmf['inference_time_ms']['mean']:.3f} ± {nmf['inference_time_ms']['std']:.3f}")

    bert = results["BERTopic"]
    print(f"{'BERTopic':<12} {bert['total_time_s']['mean']:.2f} ± {bert['total_time_s']['std']:.2f}     "
          f"{bert['peak_memory_gb']['mean']:.3f} ± {bert['peak_memory_gb']['std']:.3f}       "
          f"{bert['inference_time_ms']['mean']:.3f} ± {bert['inference_time_ms']['std']:.3f}")

    print(f"\n  Note: BERTopic embedding time (one-time): {embedding_time:.2f}s")
    print(f"        BERTopic clustering time: {bert['training_time_s']['mean']:.2f} ± {bert['training_time_s']['std']:.2f}s")

    # Save results
    results_path = RESULTS_DIR / "performance_results.json"
    print(f"\n\nSaving results to: {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone!")

    return results


if __name__ == "__main__":
    main()
