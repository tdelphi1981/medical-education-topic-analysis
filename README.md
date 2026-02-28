# Topic Analysis Experiments

Statistical topic analysis comparing LDA, NMF, and BERTopic on medical education literature.

## Setup

```bash
# Install dependencies
uv sync

# Download spaCy model
uv run python -m spacy download en_core_web_sm
```

## Running Experiments

Execute scripts in order:

```bash
# 1. Extract dataset statistics
uv run python scripts/01_data_stats.py

# 2. Preprocess corpus
uv run python scripts/02_preprocess.py

# 3. Train LDA models
uv run python scripts/03_run_lda.py

# 4. Train NMF models (requires manta-topic-modelling)
uv run python scripts/04_run_nmf.py

# 5. Train BERTopic models
uv run python scripts/05_run_bertopic.py

# 6. Evaluate and compare
uv run python scripts/06_evaluate.py

# 7. Export results to LaTeX
uv run python scripts/07_export_results.py
```

## Optional: MANTA for NMF

To use MANTA for NMF experiments:

```bash
uv pip install manta-topic-modelling
```

Note: MANTA requires scikit-learn 1.3.2, which may conflict with other dependencies.

## Output

Results are saved to:
- `outputs/models/` - Trained model files
- `outputs/results/` - JSON result files
- `outputs/latex/` - LaTeX tables and TikZ figures
