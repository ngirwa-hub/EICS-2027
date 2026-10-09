# EICS 2027: DC proposal generation, initial preparations, author reviews, and expert evaluation responses analysis

The workflow has three stages: 
    - Model-variant creation and prompting, 
    - Generated-proposal preparation, and 
    - Author consolidation, and expert-rating analysis.

## 1. Model variants and prompting

- Use the definitions in [`modelfiles/`](modelfiles/) to create the generalist, normative, and subject-matter variants in Ollama. The prompting script addresses 24 variants: two configurations per role across Llama-Pro, Mistral, Phi-4, and Gemma 3:12B.
- [`solution_propose_zeroshot.py`] queries the variants through a local Ollama service. The prompt supplies the {Anonymized} project context and 11 existing DC solutions, requesting up to five additional proposals for ports, industry, data centres, and buildings.

**Outputs:** parsed proposals in `expert_responses/solutionPropose_zeroshot.csv`, raw response logs, and an iteration counter, relative to the script's working directory. The analysis input is retained as [`solProposeZeroshot/solutionPropose_zeroshot.csv`]. The log per each iteration on each of the 24 model variants are in:  in [`solProposeZeroshot/raw_logs_solutionPropose`].

## 2. Proposal preparation and author consolidation

[`solProposeZeroshot.ipynb`](solProposeZeroshot/solProposeZeroshot.ipynb) documents the preparation workflow:

- Retain complete 50-iteration variant groups, clean titles and explanations, and retain at most five proposals per variant and iteration.
- Screen proposals against the 11 existing solutions using `BAAI/bge-large-en-v1.5` embeddings and maximum cosine similarity; retain scores below `0.60`.
- Cluster retained proposals using HDBSCAN, label and summarise the eight non-noise clusters, and consolidate their themes into five authored proposals in [`author-final.md`](solProposeZeroshot/author-final.md).
- Assess correspondence with source-cluster evidence using BERTScore (`roberta-large`) in [`clusterAuthorSemFidelityCheck.py`](solProposeZeroshot/clusterAuthorSemFidelityCheck.py).

**Outputs:** intermediate cleaning files, semantic-screening results, cluster assignments and summaries, the five authored proposals, [semantic-fidelity results](solProposeZeroshot/semantic-fidelity/), and the Latex table: [cluster-to-proposal appendix table](solProposeZeroshot/outputs/tables/cluster_proposal_compilation.tex).

## 3. Expert-rating analysis

**Evaluation materials:**

- [Survey consent](<expertsRatings/Survey Consent.pdf>).
- [Aware-condition evaluation form](<expertsRatings/Assessment of Novelty of DC Solutions.pdf>).
- [Blind-condition evaluation form](<expertsRatings/Novelty of DC Solutions.pdf>).

[`expertsRatings.ipynb`](expertsRatings/expertsRatings.ipynb) analyses the aware and blind survey workbooks: 13 consenting experts that responded to the 'DC proposal evaluation forms', comprising five aware and eight blind experts.

- Prepare anonymous expert profiles and proposal-level judgeability data. Present unable-to-judge percentages alongside selected written reasons; rating analyses include only experts able to judge the corresponding proposal. Outputs saved in: `/Users/HP/Documents/EICS-2026/expertsRatings/outputs/tables/`for detailed table is: `expert_detailed_profile.tex` and summarized content is: `expert_profile_table.tex`
- Describe novelty drivers and previous exposure using nominal response distributions. Summarise ordinal novelty, perceived difference, and confidence ratings using response distributions, median categories, and quartile categories. Outputs saved in: `/Users/HP/Documents/EICS-2026/expertsRatings/outputs/figures/`: nominal responses in [proposal_nominal_responses.eps](expertsRatings/outputs/figures/proposal_nominal_responses.eps) and ordinal ratings in [proposal_ordinal_ratings.eps](expertsRatings/outputs/figures/proposal_ordinal_ratings.eps). Both figures are also available in PNG format.
- Assess novelty–difference associations using proposal-specific Kendall's tau-b, pooling conditions and retaining complete pairs from the same expert, with bootstrap confidence intervals and permutation p-values. Output saved in: `/Users/HP/Documents/EICS-2026/expertsRatings/outputs/tables/`: [kendall_tau_novelty_difference.tex](expertsRatings/outputs/tables/kendall_tau_novelty_difference.tex).
- Compare aware and blind ratings separately for each proposal and outcome using Mann–Whitney U, exact permutation p-values, and rank-biserial effects with bootstrap confidence intervals. Output saved in: `/Users/HP/Documents/EICS-2026/expertsRatings/outputs/tables/`: [aware_blind_proposal_comparisons.tex](expertsRatings/outputs/tables/aware_blind_proposal_comparisons.tex).

**Multiplicity:** Holm correction covers the five Kendall tests and, separately, the five Mann–Whitney tests within each outcome. Bootstrap confidence intervals are individual 95% intervals, not multiplicity-adjusted.

**Outputs:** publication figures in [`outputs/figures/`](expertsRatings/outputs/figures/) and LaTeX profile, written-reason, Kendall, and aware–blind comparison tables in [`outputs/tables/`](expertsRatings/outputs/tables/).
