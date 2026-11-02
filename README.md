# EICS 2027: DC proposal generation, preparation, consolidation, and expert evaluation

The workflow has three stages:

1. Model-variant creation and prompting.
2. Generated-proposal preparation, semantic-diversity analysis, OpenAI cleaning, and author consolidation.
3. Expert-rating analysis.

The proposal workflow proceeds from **cluster summaries → eight OpenAI-cleaned proposals → five author-consolidated proposals**, with BERTScore comparisons between the successive versions.

## 1. Model variants and prompting

- Use the definitions in [`modelfiles/`](modelfiles/) to create the generalist, normative, and subject-matter variants in Ollama. The prompting script addresses 24 variants: two configurations per role across Llama-Pro, Mistral, Phi-4, and Gemma 3:12B.
- [`solution_propose_zeroshot.py`](promptScripts/solution_propose_zeroshot.py) queries the variants through a local Ollama service. The prompt supplies the {Anonymized} project context and 11 existing DC solutions, requesting up to five additional proposals for ports, industry, data centres, and buildings.

**Outputs:** parsed proposals in `expert_responses/solutionPropose_zeroshot.csv`, raw response logs, and an iteration counter, relative to the script's working directory. The analysis input is retained as [`solutionPropose_zeroshot.csv`](solProposeZeroshot/solutionPropose_zeroshot.csv). Archived logs for the model variants and iterations are in [`raw_logs_solutionPropose/`](solProposeZeroshot/raw_logs_solutionPropose/).

## 2. Proposal preparation and author consolidation

[`solProposeZeroshot.ipynb`](solProposeZeroshot/solProposeZeroshot.ipynb) documents the preparation workflow:

- Retain complete 50-iteration variant groups, clean titles and explanations, and retain at most five proposals per variant and iteration.
- Screen proposals against the 11 existing solutions using `BAAI/bge-large-en-v1.5` embeddings and maximum cosine similarity; retain scores below `0.60`.
- Cluster retained proposals using HDBSCAN and prepare labels, key phrases, summaries, titles, and explanations in [`cluster_summaries_zeroshot.csv`](solProposeZeroshot/clustering_zeroshot/cluster_summaries_zeroshot.csv).
- Use [`generation/workflow.py`](solProposeZeroshot/computational-proposals/generation/workflow.py) with [`instructions.md`](solProposeZeroshot/computational-proposals/generation/instructions.md) to structure the eight non-noise clusters into [`eight-proposals.md`](solProposeZeroshot/computational-proposals/generation/eight-proposals.md). Each cleaned proposal represents one cluster and contains a name, key phrases, objective, and description. 
- Consolidate related OpenAI-cleaned proposals into five final proposals in [`author-final.md`](solProposeZeroshot/author-final.md), retaining their source-cluster memberships.
- Generate the [proposal-mapping table](solProposeZeroshot/computational-proposals/tables/cluster_proposal_compilation.tex), showing source-cluster IDs, OpenAI-cleaned proposal names, author-consolidated proposal names, and final proposal IDs. Merged cells identify combined proposals.
- Evaluate semantic correspondence using [`evaluation/compare_bertscore.py`](solProposeZeroshot/computational-proposals/evaluation/compare_bertscore.py), with both comparisons stored in [`bertscore-comparisons.csv`](solProposeZeroshot/computational-proposals/evaluation/bertscore-comparisons.csv):

| Comparison | Reference | Candidate | Alignment |
| --- | --- | --- | --- |
| Source clusters → OpenAI | Cluster-summary CSV | Eight OpenAI-cleaned proposals | One proposal per source cluster |
| OpenAI → author | OpenAI-cleaned proposals | Five author-consolidated proposals | Concatenate OpenAI content according to each final proposal's source-cluster membership |

Both comparisons score key phrases separately from combined objectives and descriptions. For the source comparison, the references are `sample_titles + summary` and `explanations_text`, respectively. BERTScore uses `roberta-large`, without IDF weighting or baseline rescaling. The CSV contains 26 individual comparisons and four unweighted means. BERTScore measures semantic correspondence.

Generation scripts, instructions, generated proposals, evaluation files, and the mapping table are grouped under [`computational-proposals/`](solProposeZeroshot/computational-proposals/). Source evidence remains under `clustering_zeroshot/`, and the final authored proposals remain in `author-final.md`.

### Semantic diversity across model variants and temperatures

[`solProposeZeroshot.ipynb`](solProposeZeroshot/solProposeZeroshot.ipynb) also compares the semantic content of the model variants' outputs. Using all cleaned proposals before similarity screening, it combines proposal titles and available explanations into one text per variant and iteration: 1,200 outputs across 24 variants and 50 iterations. Texts are embedded using `BAAI/bge-large-en-v1.5`. Each variant comparison reports the mean cosine distance across all 50 × 50 output pairs.

The role-specific temperature pairs are Generalist: `0.75`/`0.80`, Normative: `0.85`/`0.90`, and Subject-matter: `0.65`/`0.70`.

| Comparison | Role | Temperature | Number of comparisons |
| --- | --- | --- | --- |
| Within each model | Same | Different | 12 |
| Between models | Same | Same | 36 |
| Between models | Same | Different, using both temperature assignments | 36 |

The analysis contains 84 comparisons in total. The [saved distance CSV](solProposeZeroshot/computational-proposals/semantic-distance/same_role_distances.csv) contains the first 48; the additional 36 cross-model, cross-temperature comparisons are displayed in the notebook. Heatmaps show [within-model and same-temperature cross-model distances](solProposeZeroshot/computational-proposals/semantic-distance/same_role_distances.png) and [different-temperature cross-model distances](solProposeZeroshot/computational-proposals/semantic-distance/same_role_cross_temperature_distances.png). Both figures are available in EPS, PDF, and PNG formats under [`semantic-distance/`](solProposeZeroshot/computational-proposals/semantic-distance/).

## 3. Expert-rating analysis

**Evaluation materials:**

- [Survey consent](<expertsRatings/Survey Consent.pdf>).
- [Aware-condition evaluation form](<expertsRatings/Assessment of Novelty of DC Solutions.pdf>).
- [Blind-condition evaluation form](<expertsRatings/Novelty of DC Solutions.pdf>).

[`expertsRatings.ipynb`](expertsRatings/expertsRatings.ipynb) analyses the aware and blind survey workbooks: 13 consenting experts that responded to the 'DC proposal evaluation forms', comprising five aware and eight blind experts.

- Prepare anonymous expert profiles and proposal-level judgeability data. Present unable-to-judge percentages alongside selected written reasons; rating analyses include only experts able to judge the corresponding proposal. Profile tables: [detailed profiles](expertsRatings/outputs/tables/expert_detailed_profile.tex) and [profile summary](expertsRatings/outputs/tables/expert_profile_table.tex).
- Extract additional free comments on P1–P5 into a [LaTeX comment table](expertsRatings/outputs/tables/expert_additional_comments.tex), containing 24 non-empty entries from nine experts. This extraction applies no judgeability filtering, omits blank fields, retains responses such as “no comments,” and preserves respondents' wording and spelling. Entries are identified by proposal and anonymous expert code; the table notes the four experts who left all five comment fields blank.
- Describe novelty drivers and previous exposure using nominal response distributions. Summarise ordinal novelty, perceived difference, and confidence ratings using response distributions, median categories, and quartile categories. Figures: [nominal responses](expertsRatings/outputs/figures/proposal_nominal_responses.eps) and [ordinal ratings](expertsRatings/outputs/figures/proposal_ordinal_ratings.eps). Both figures are also available in PNG format.
- Assess novelty–difference associations using proposal-specific Kendall's tau-b, pooling conditions and retaining complete pairs from the same expert, with bootstrap confidence intervals and permutation p-values. Table: [Kendall associations](expertsRatings/outputs/tables/kendall_tau_novelty_difference.tex).
- Compare aware and blind ratings separately for each proposal and outcome using Mann–Whitney U, exact permutation p-values, and rank-biserial effects with bootstrap confidence intervals. Table: [aware–blind comparisons](expertsRatings/outputs/tables/aware_blind_proposal_comparisons.tex).

**Multiplicity:** Holm correction covers the five Kendall tests and, separately, the five Mann–Whitney tests within each outcome. Bootstrap confidence intervals are individual 95% intervals, not multiplicity-adjusted.

**Outputs:** publication figures in [`outputs/figures/`](expertsRatings/outputs/figures/) and LaTeX profile, written-reason, additional-comment, Kendall, and aware–blind comparison tables in [`outputs/tables/`](expertsRatings/outputs/tables/).
