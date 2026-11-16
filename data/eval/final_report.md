# RAG Retrieval Benchmark — Final Technical Report

## 1. Project Overview

This project evaluates three information retrieval approaches for a
legal-domain Retrieval-Augmented Generation (RAG) system:

- Dense Retrieval
- BM25 Sparse Retrieval
- Hybrid Retrieval using Reciprocal Rank Fusion (RRF)

The objective is to compare retrieval quality using a fixed corpus,
a manually prepared ground-truth dataset, and reproducible experiments.

This project evaluates the retrieval stage. It does not claim to evaluate
the quality or factual correctness of final generated answers.

## 2. Dataset and Corpus

| Property | Value |
|---|---:|
| Number of source documents | 20 |
| Number of corpus chunks | 5,782 |
| Embedding model | BGE-M3 |
| Embedding dimension | 1,024 |
| Embedding format | NumPy float32 |
| Ground-truth questions | 100 |
| Answerable questions | 90 |
| Unanswerable questions | 10 |
| Missing chunk references | 0 |

The corpus and embedding data were validated for structural consistency.
All ground-truth relevant chunk identifiers were found in the corpus.

The existing embeddings were reused. They were not recomputed during
the benchmark experiments.

## 3. Ground-Truth Dataset

The evaluation dataset contains 100 questions distributed across
the following categories:

| Category | Number of questions |
|---|---:|
| Simple | 40 |
| Paraphrase | 15 |
| Multi-hop | 10 |
| Legal terminology | 15 |
| Similar documents | 10 |
| Unanswerable | 10 |
| **Total** | **100** |

The dataset includes both answerable and unanswerable questions.
Structural validation confirmed that all referenced relevant chunks
exist in the corpus.

## 4. Retrieval Methods

### 4.1 Dense Retrieval

Dense retrieval represents queries and corpus chunks as vectors
and ranks candidate chunks using vector similarity.

The project uses precomputed BGE-M3 embeddings.

### 4.2 BM25 Retrieval

BM25 is a sparse lexical retrieval method. It ranks documents based
on query-term matching and term-frequency statistics.

### 4.3 Hybrid Retrieval

Hybrid retrieval combines Dense and BM25 rankings using
Reciprocal Rank Fusion (RRF).

The baseline experiment uses RRF with k = 60.
Additional experiments evaluate k = 20 and a weighted
Dense-to-BM25 configuration of 2:1.

## 5. Evaluation Metrics

The benchmark reports:

- Hit@1: whether a relevant chunk appears at rank 1.
- Hit@3: whether a relevant chunk appears in the first 3 results.
- Hit@5: whether a relevant chunk appears in the first 5 results.
- Recall@5: proportion of relevant chunks retrieved within the first 5.
- MRR: Mean Reciprocal Rank of the first relevant result.
- NDCG@5: ranking quality with position-sensitive relevance.

The reported metrics are retrieval metrics, not answer-generation metrics.

## 6. Baseline Results

| Metric | Dense | BM25 | Hybrid RRF |
|---|---:|---:|---:|
| Hit@1 | 0.7778 | 0.6667 | **0.7889** |
| Hit@3 | 0.9778 | 0.9111 | 0.9667 |
| Hit@5 | **0.9889** | 0.9444 | 0.9778 |
| Recall@5 | **0.9389** | 0.8722 | 0.9222 |
| MRR | 0.8744 | 0.7819 | 0.8744 |
| NDCG@5 | 0.8595 | 0.7755 | **0.8609** |

Dense retrieval achieves the strongest Hit@5 and Recall@5 results.
Hybrid retrieval achieves a slightly higher Hit@1 and NDCG@5.
Dense and Hybrid have the same MRR in the baseline experiment.
BM25 performs below the other two methods on the reported metrics.

## 7. Parameter Experiments

### 7.1 RRF k = 20

| Metric | Hybrid RRF k=20 |
|---|---:|
| Hit@1 | 0.7889 |
| Hit@5 | 0.9778 |
| Recall@5 | 0.9278 |
| MRR | 0.8744 |
| NDCG@5 | 0.8636 |

Compared with baseline Hybrid RRF, k=20 produces a small improvement
in Recall@5 and NDCG@5. Hit@1, Hit@5, and MRR remain unchanged.

### 7.2 Top-K = 10

The top-K experiment completed successfully.
The currently reported summary metrics are still calculated at
the first five positions. Therefore, these values alone cannot
establish the effect of retrieving ten results. A dedicated
evaluation at K=10 is needed for that comparison.

### 7.3 Weighted RRF: Dense 2, BM25 1

| Metric | Weighted Hybrid |
|---|---:|
| Hit@1 | 0.7556 |
| Hit@5 | 0.9778 |
| Recall@5 | 0.9222 |
| MRR | 0.8578 |
| NDCG@5 | 0.8505 |

The tested 2:1 weighting does not outperform the baseline Hybrid
configuration on the reported metrics.

## 8. Error Analysis

The baseline result contains 40 failure records across retrieval
methods. These records correspond to 19 unique questions.

Failure records by method:

| Method | Failure records |
|---|---:|
| Dense | 10 |
| BM25 | 18 |
| Hybrid RRF | 12 |
| **Total** | **40** |

Failure records by category:

| Category | Failure records |
|---|---:|
| Legal terminology | 3 |
| Simple | 10 |
| Paraphrase | 4 |
| Multi-hop | 15 |
| Similar documents | 8 |
| **Total** | **40** |

Failure classification:

| Failure type | Records |
|---|---:|
| No relevant chunk in top 5 | 8 |
| Relevant chunk ranked too low or missing | 32 |
| **Total** | **40** |

Multi-hop questions account for a notable share of the observed
failure records. This suggests that questions requiring information
from multiple pieces of text deserve further investigation.

### Example: q004

Question: "در قانون مدنی، عقد چگونه تعریف شده است؟"

Expected relevant chunk: `civil_law-0080`

The referenced corpus text contains the legal definition of a contract
from Article 183 of the Civil Code. The reference is valid, but the
expected chunk was not retrieved in the Dense top five results for
this query.

This is a retrieval failure rather than a missing ground-truth reference.

## 9. Automated Validation and Tests

The final audit confirmed:

- 100 ground-truth questions.
- 90 answerable and 10 unanswerable questions.
- 5,782 corpus chunks.
- Zero missing relevant chunk references.
- 40 baseline failure records.

The automated test suite completed successfully:
15 tests passed.

## 10. Limitations

1. The benchmark evaluates retrieval quality, not end-to-end generated
   answer quality.
2. The ground-truth dataset contains 100 questions and may not cover
   every legal topic or query formulation.
3. The reported top-K experiment currently uses metrics at K=5.
   A separate K=10 metric calculation is required.
4. The differences between some retrieval configurations are small
   and should not be interpreted as statistically significant without
   additional testing.
5. The existing BGE-M3 embeddings were reused rather than recomputed.
6. Failure root causes require further manual inspection before
   assigning definitive explanations.

## 11. Conclusion

Dense retrieval provides the strongest overall baseline performance
for Hit@5 and Recall@5 in this dataset. Hybrid RRF offers a small
advantage in Hit@1 and NDCG@5, while BM25 alone performs less strongly.

The RRF k=20 configuration gives a modest improvement in Hybrid
Recall@5 and NDCG@5 compared with the baseline Hybrid configuration.
The tested weighted 2:1 configuration does not provide an improvement.

Based on the current experiments, Dense retrieval is the strongest
baseline choice when the primary objective is retrieving relevant
chunks within the top five results. Hybrid retrieval remains a useful
alternative when first-result ranking and ranking quality are also
important.

Further work should focus on multi-hop retrieval failures, query
expansion, reranking, and a dedicated top-10 evaluation.

## 12. Reproducibility

The repository contains the retrieval implementations, evaluation
code, test suite, configuration files, ground-truth dataset,
corpus and experiment outputs.

The test suite can be executed using:

`python -m pytest -q`

The experiment runner can be executed using:

`python scripts/run_experiments.py`
