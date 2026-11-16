# Delivery status

## Ready in this package
- 100 Ground Truth questions (90 answerable, 10 unanswerable) with 6 required categories.
- Evidence IDs checked against the 5,782-chunk corpus; `expected_documents` cross-checked against chunk metadata.
- Dense, BM25, Hybrid RRF implementations.
- BM25 fallback for offline environments where `rank_bm25` is not installed.
- Six ranking metrics and category-level summaries.
- Unit tests independent of Ollama.
- Baseline and one-factor experiment configurations.
- 15 archived failure cases from the previous 20-question run.
- Offline BM25 smoke result on all 100 questions.
- README, metric notes, ground-truth review, and engineering analysis.

## Verified in the package-building environment
- `python -m pytest`: **15 passed**
- `python scripts/validate_ground_truth.py`: **100 questions; VALIDATION PASSED**
- Python syntax compilation: passed
- Offline BM25 fallback smoke: completed on 100 questions

## Still requires the user's local Ollama service
The complete 100-question Dense/BM25/Hybrid benchmark and the new parameter experiments could not be executed in this environment because the local Ollama service is not available here. Run the following in the existing project directory:

```powershell
pip install -r requirements.txt
python scripts/validate_ground_truth.py
pytest
python scripts/run_experiments.py
python scripts/run_benchmark.py --config configs/weighted_rrf.yaml
```

Before the benchmark, confirm `ollama list` includes `bge-m3` and Ollama is listening at `http://localhost:11434`. Keep the existing `data/embeddings/dense_embeddings.npy`; do not recompute embeddings unless you intentionally rebuild the corpus with the same embedding model.

## Final Git commit
The uploaded ZIP did not contain the `.git` directory, so no commit can be created from this extracted copy. Unzip the updated files over the existing local repository (do not delete its `.git` directory), then:

```powershell
git status
git add .
git commit -m "Complete advanced retrieval benchmark"
git log -1 --oneline
```

Do not submit the archived 20-question metrics as if they were the final 100-question three-method result. The new full result files will be written to `results/` after the local run.
