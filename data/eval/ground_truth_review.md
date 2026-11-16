# Ground Truth review

## Snapshot
- Total questions: 100
- Answerable: 90
- Unanswerable: 10
- Corpus: 5,782 chunks across 20 documents
- All referenced chunk IDs were checked against `chunks.jsonl`.
- For each answerable record, `expected_documents` is derived from the document IDs of its cited chunks.

## Category distribution
| Category | Count |
|---|---:|
| simple | 40 |
| paraphrase | 15 |
| multi_hop | 10 |
| legal_terminology | 15 |
| similar_documents | 10 |
| unanswerable | 10 |

## Evidence procedure
1. Read the exact text of every cited chunk for q001-q036 and the newly added q037-q090 records.
2. Keep the question only when its answer is explicitly stated or can be assembled from the cited chunks.
3. For multi-hop questions, cite the chunks containing the distinct pieces of evidence.
4. Derive `expected_documents` from `metadata.document_id`; do not type document labels independently.
5. For q091-q100, use empty document/chunk lists and mark `answerable=false`; these are deliberately outside the 20-document corpus.
6. Run `python scripts/validate_ground_truth.py` and the unit test suite after editing.

## Important limitation
The validator proves schema correctness, unique IDs, and that references exist and belong to the declared documents. It cannot prove semantic relevance. The evidence review above is a manual content check of the cited passages, but the ten unanswerable items should still be spot-checked by a second reviewer before submission. A corpus-wide absence claim is harder to prove than a positive citation.

## Unanswerable policy
Unanswerable questions are excluded from Hit@K, Recall@K, MRR, and nDCG averages because these ranking metrics require at least one relevant chunk. Their count is reported separately. A production RAG system should add a calibrated abstention threshold and measure false-answer / abstention behavior; this benchmark currently measures retrieval ranking, not answer-generation abstention.
