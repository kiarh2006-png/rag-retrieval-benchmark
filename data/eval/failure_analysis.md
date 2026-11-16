# Retrieval Failure Analysis

## 1. Overview

The failure inspection examined 16 representative questions using the actual corpus chunk text and the top-5 retrieved results from Dense, BM25, and Hybrid RRF.

The inspection identified four main sources of retrieval errors:

1. Relevant information distributed across adjacent chunks.
2. Partial retrieval in multi-hop questions.
3. Lexical or semantic ranking mismatches.
4. Potential ground-truth and chunk-boundary inconsistencies.

A question was not considered fully retrieved merely because one of its expected chunks appeared in the results. Questions requiring multiple chunks were evaluated as incomplete when one or more required chunks were missing.

## 2. Case-by-Case Analysis

### q001 — Article 154: Theft conditions

- Expected chunks: `taazirat-0130`, `taazirat-0131`
- Dense retrieves `taazirat-0131`, but misses `0130`.
- BM25 and Hybrid retrieve `0130`, but miss `0131`.
- Root cause: The article introduction and its five legal conditions are split across adjacent chunks.
- Failure type: Partial multi-chunk retrieval.
- Suggested improvement: Preserve adjacent legal provisions as contextual candidates and evaluate complete article coverage.

### q002 — Article 154: Scenario-based theft question

- Expected chunks: `taazirat-0130`, `taazirat-0131`
- All three methods retrieve `taazirat-0131`.
- The article introduction and penalty information in `0130` are not retrieved.
- Root cause: The query strongly matches the numbered conditions, while the article introduction is less directly matched.
- Failure type: Partial retrieval and ranking mismatch.
- Suggested improvement: Use neighboring-chunk expansion for legal questions.

### q004 — Civil Code: Definition of contract

- Expected chunk: `civil_law-0080`
- Dense and Hybrid retrieve the adjacent chunk `civil_law-0081`.
- BM25 retrieves unrelated chunks.
- Inspection shows that `0080` contains the definition of a contract, while `0081` repeats part of the sentence at the chunk boundary.
- Root cause: Overlapping text and chunk-boundary ambiguity, combined with lexical retrieval failure.
- Failure type: Boundary-related relevance and ranking mismatch.
- Suggested improvement: Review the ground-truth reference and avoid penalizing retrieval of an adjacent chunk when it contains the same relevant passage.

### q008 — Article 183: Definition of contract

- Expected chunk: `civil_law-0080`
- BM25 does not retrieve the expected chunk in its top five.
- The retrieved results include another legal provision containing the same article number in a different law.
- Root cause: Article-number ambiguity across multiple legal documents.
- Failure type: Lexical ambiguity and ranking mismatch.
- Suggested improvement: Include document identity and legal context in query processing.

### q015 — Article 193: Transaction by conduct

- Expected chunk: `civil_law-0083`
- BM25 does not retrieve the expected chunk in its top five.
- The expected chunk explicitly contains Article 193 and explains transaction formation through conduct.
- Root cause: BM25 fails to rank the relevant legal passage highly enough for this wording.
- Failure type: Lexical retrieval and ranking failure.
- Suggested improvement: Combine lexical retrieval with semantic retrieval and evaluate query expansion for legal terminology.

### q018 — Civil Procedure Article 2

- Expected chunks: `civil_procedure-0000`, `civil_procedure-0001`
- Dense and Hybrid retrieve `0000`, but miss `0001`.
- BM25 retrieves `0000`, but not `0001`.
- The second chunk completes the list of persons authorized to request judicial proceedings.
- Root cause: The legal sentence is split across adjacent chunks.
- Failure type: Partial multi-chunk retrieval.
- Suggested improvement: Apply neighboring-chunk expansion and evaluate complete passage coverage.

### q053 — Historical meaning of the word Computer

- Expected chunk: `computers-0011`
- BM25 retrieves `computers-0014` instead.
- The expected chunk contains the historical meaning of computer as a person who performs calculations.
- Root cause: The query shares terminology with several chunks, causing an incorrect ranking.
- Failure type: Lexical ranking mismatch.
- Suggested improvement: Improve phrase-level matching and use semantic ranking to distinguish historical definitions from modern usage.

### q060 — Public and private claims in criminal procedure

- Expected chunks: `criminal_procedure-0001`, `criminal_procedure-0002`
- BM25 retrieves `0001`, but does not retrieve `0002`.
- The second chunk contains the detailed explanation of public and private claims.
- Root cause: The question requires information distributed across two consecutive chunks.
- Failure type: Partial multi-hop retrieval.
- Suggested improvement: Expand retrieved legal passages using adjacent chunk relationships.

### q061 — Taxpayers and exemptions

- Expected chunks: `direct_taxes-0000`, `direct_taxes-0001`
- Dense retrieves `0000`, but misses `0001`.
- BM25 and Hybrid fail to retrieve the expected chunks in their top five.
- The question requires both the groups subject to tax and the groups exempt from tax.
- Root cause: Multi-part information is distributed across adjacent legal provisions.
- Failure type: Multi-hop retrieval failure.
- Suggested improvement: Improve retrieval for queries that combine taxpayer categories and exemptions; evaluate query decomposition.

### q062 — Worker and employer definitions

- Expected chunks: `labor_law-0000`, `labor_law-0001`
- Dense retrieves `0000`, but misses `0001`.
- BM25 retrieves `0000`, but misses the continuation in `0001`.
- The employer definition continues across the chunk boundary.
- Root cause: Incomplete retrieval of a split legal definition.
- Failure type: Partial retrieval.
- Suggested improvement: Preserve the continuity of definitions and retrieve neighboring chunks when a sentence is incomplete.

### q063 — Electronic data messages

- Expected chunks: `ecommerce_law-0000`, `ecommerce_law-0019`
- BM25 retrieves `0019`, but misses `0000`.
- The two expected chunks describe different parts of the question: the definition of a data message and its evidentiary status.
- Root cause: The question combines two distinct concepts from different parts of the same document.
- Failure type: Multi-hop retrieval and query decomposition issue.
- Suggested improvement: Split compound queries into semantic sub-questions and merge their retrieved candidates.

### q064 — Divorce counseling and arbitrator appointment

- Expected chunks: `family_protection-0028`, `family_protection-0031`
- Dense and Hybrid retrieve `0028`, but miss `0031`.
- The second chunk contains the court's authority to appoint an arbitrator if the parties cannot introduce one.
- Root cause: The question combines counseling procedures with a separate arbitration provision.
- Failure type: Multi-hop retrieval failure.
- Suggested improvement: Use query decomposition and candidate fusion for questions involving separate legal procedures.

### q066 — Tehran's economic share and GDP composition

- Expected chunks: `iran_economy-0001`, `iran_economy-0002`
- All three methods retrieve `0001`, but miss `0002`.
- The second chunk contains the sectoral composition of GDP.
- Root cause: The query contains two related but distinct information requirements.
- Failure type: Partial multi-hop retrieval.
- Suggested improvement: Decompose the query into regional GDP share and sectoral composition sub-queries.

### q067 — Iranian music performance and Elam carving

- Expected chunks: `persian_music-0009`, `persian_music-0017`
- BM25 and Hybrid retrieve `0009`, but miss `0017`.
- The expected chunks discuss different subjects: the musicians depicted in the Elam carving and the conventional structure of a performance.
- Root cause: Two separate passages are needed to answer the compound question.
- Failure type: Multi-hop retrieval failure.
- Suggested improvement: Use query decomposition and combine candidates from separate semantic searches.

### q083 — Legal definitions of worker and employer

- Expected chunks: `labor_law-0000`, `labor_law-0001`
- Dense retrieves `0000`, but misses `0001`.
- BM25 retrieves `0001`, but misses `0000`.
- Hybrid retrieves `0000`, but misses `0001`.
- Root cause: The relevant definitions are split across adjacent chunks, and different methods retrieve different parts.
- Failure type: Partial retrieval and chunk-boundary issue.
- Suggested improvement: Add adjacent-chunk expansion and evaluate document-level coverage.

### q085 — Public and private aspects of criminal offenses

- Expected chunks: `criminal_procedure-0001`, `criminal_procedure-0002`
- All three methods retrieve `0002`, but miss `0001`.
- The second chunk already contains the key explanation of public and private claims and their prosecution.
- Root cause: The expected reference may be stricter than necessary for this question.
- Failure type: Potential ground-truth over-specification.
- Suggested improvement: Review whether `criminal_procedure-0002` alone is sufficient as the expected relevant chunk.

## 3. Cross-Method Comparison

The inspected cases show that the retrieval methods fail in different ways:

- Dense retrieval often identifies semantically related passages but can miss a required neighboring chunk.
- BM25 performs well when query terms closely match the source, but can rank passages from unrelated documents or nearby concepts too highly.
- Hybrid RRF combines lexical and semantic rankings, but it cannot guarantee complete retrieval when the question requires multiple separated passages.
- No method consistently retrieves every required chunk across all inspected multi-part questions.

The q001 and q083 cases demonstrate that different retrieval methods can retrieve complementary parts of the same answer. This supports evaluating candidate fusion and contextual expansion rather than relying exclusively on a single retrieval method.

## 4. Recommended Improvements

The inspection suggests the following improvements for future experiments:

1. Adjacent-chunk expansion: Retrieve neighboring chunks when a relevant chunk appears to contain an incomplete sentence or provision.
2. Query decomposition: Split compound questions into smaller information requirements before retrieval.
3. Ground-truth review: Reassess references where overlapping chunks contain the same relevant passage or where one chunk may independently answer the question.
4. Document-aware retrieval: Use document identity and legal context to reduce ambiguity between identical article numbers in different laws.
5. Reranking: Apply a second-stage ranking model to the candidate set when lexical and semantic retrieval disagree.
6. Multi-hop evaluation: Report both whether at least one relevant chunk was found and whether all required chunks were retrieved.

## 5. Limitations

This inspection is based on representative failure candidates and top-five retrieval results. It does not establish that every proposed improvement will increase benchmark performance.

Ground-truth changes should only be made after reviewing the original question and source passage. The observed retrieval failures and possible annotation issues must remain separate in the final evaluation.

## 6. Conclusion

The main limitation observed in this benchmark is not simply the inability to retrieve relevant documents. Several failures arise because the required information is distributed across multiple chunks, while the evaluation expects all of them to appear within the top-five results.

The findings support future experiments with contextual chunk expansion, query decomposition, document-aware retrieval, and ground-truth refinement.