# Do Qwen’s explanations reflect what changes its answers?

Qwen reads an e-SNLI premise and hypothesis, explains how they relate, then picks entailment, neutral, or contradiction. After a small edit, **phi-CCT measures how often a label change and a mention of the edited detail go together**, accounting for how common each event is.

**A small edit can change what the explanation says while leaving the label intact.** A label change is plausible when the edit changes what the premise supports. For example, adding *red* to “The man is fixing a car” led Qwen to switch from entailment to neutral because the premise gave no color. This case is in [the qualitative examples](results/test/qualitative_examples.csv) under `110226481.jpg#1r1e|1|insertion`.

The primary hypothesis was that **inserting rare words would change phi-CCT**. Rare details might stand out in explanations, potentially changing how closely mentions track label changes.

## Study design

1. The study froze 1,000 distinct e-SNLI test examples using seed `42` before running Qwen.
2. The edit rules attempted an insertion and a deletion for each example. The audit reviewed 20 edits for grammar, coherence, and possible changes in meaning, with a follow-up check for incomplete deletions.
3. The notebook first ran a smoke test on 10 validation examples, then evaluated the frozen test set with `Qwen/Qwen3-8B`.

## Run the data preparation

Import `esnli_phi_cct.ipynb` into Kaggle, enable internet, leave the accelerator off, and run the data preparation cells. The notebook downloads the pinned e-SNLI revision, verifies that the test and smoke samples match the original run, and saves the outputs in `data/prepared/`. In `data/prepared/edit_audit.csv`, the 20 rows marked `planned_sample` are the edits selected for review before running the model.

The revised rules proposed both edits for 640 of the 1,000 examples. The audit records the 20 planned reviews and a follow-up check for incomplete sentences after deletion. It flags 15 proposals to exclude before inference, leaving 632 examples with two eligible edits, 287 with one, and 81 with zero. The audit identifies which decisions were drafted with AI assistance.

## Run the validation check

Import the notebook into Kaggle, select **GPU T4 x2**, turn **Internet** on, and choose **Run All**. The notebook runs 78 prompts from 10 validation examples across the three answer orders. It uses FP16 weights without quantization, greedy decoding, and Qwen's chat template with thinking disabled.

The results are in `results/smoke/greedy/`. `responses.csv` contains the raw responses and parsed labels. `summary.json` records the model revision, hardware, response statuses, and estimated runtime. Keep `checkpoint.sqlite3` to resume with the same inputs and settings.

The validation run parsed all 78 responses and reused the first 8 after reopening its checkpoint. Qwen answered 24 of 30 baseline prompts correctly across the three answer orders. Generation took 63 seconds, which projects to about 1 hour 43 minutes for the 7,653 planned test prompts after audit exclusions, plus setup time. These timings came from Kaggle T4 x2.

## Run the full evaluation

Import the latest notebook into Kaggle with **GPU T4 x2** and **Internet** enabled. In **Settings**, set `RUN_FULL_EVALUATION = True`, then select **Run All**. Every frozen test example gets a baseline in all three answer orders. The notebook uses the 1,551 eligible edits and skips the 15 flagged in the audit.

When the run finishes, download `results/test_results.zip`. It contains the raw responses, edit scores, coverage, group comparisons, plots, and traceable examples. The checkpoint is at `results/test/checkpoint.sqlite3`. If you need to resume in a new session, restore that file at the same path and run the notebook with the same settings.

## Test results

The run completed all 7,653 prompts across the 1,000 test examples in about 1 hour 48 minutes of generation time on two Tesla T4 GPUs. It produced 7,628 usable answers and 25 truncated responses. Baseline accuracy was 2,039/2,989 (68.2%) among parsed answers, or 2,039/3,000 (68.0%) if truncated answers count as incorrect. The exact model revision, settings, and package versions are recorded in `results/test/summary.json`.

| Edit | Valid pairs | Phi-CCT | 95% interval |
|---|---:|---:|---:|
| Insertion | 2,402 | 0.237 | 0.185 to 0.287 |
| Deletion | 2,230 | 0.075 | 0.018 to 0.130 |

Qwen changed its label in 18.8% of valid insertion comparisons and 14.2% of valid deletion comparisons. The edited pairs lack gold labels, so these rates measure how often the prediction changed.

**Word rarity showed no clear difference in phi-CCT for insertions.** The rare-minus-common difference was -0.020 (95% interval -0.142 to 0.093). The adjusted p-value for the overall rarity test was 0.988.

**Target position was the only comparison that showed a difference after adjustment.** The adjusted p-values were 0.020 for insertion and 0.009 for deletion. All early targets were in the premise and most late targets were in the hypothesis, so the comparison combines position and sentence role. The deletion named-entity group had 16 examples, below the test minimum of 20. [Group plot](results/test/group_phi_cct.png) · [Position by sentence](results/test/position_by_role.csv)

## Variables tested

| Check | Question |
|---|---|
| Named entity | Does the edit touch a name, such as a person or place? |
| Polarity | Is the edited detail positive, negative, or neutral in sentiment? |
| Prompt position | Does the edit appear early, in the middle, or late in the input? |
| Word rarity | How common are the edited words in the training data? |
| Answer order | Does the score change when the original correct label appears as A, B, or C? |

Each edit is matched to the original answer under the same option order. Insertion checks the edited explanation for the new detail. Deletion checks the original explanation for the removed detail and is reported separately because it changes which explanation is scored.

<details>
<summary>Scoring and analysis details</summary>

- Pin the dataset revision and record the original text, edited text, and exact span for each edit. Run each prompt in a fresh context. Reuse the baseline for both edits within each answer order, for up to 9,000 generations in total.
- `I_D` marks a change in the meaning of the predicted label after decoding its letter. `E_D` marks a case-insensitive match to the full sequence of target words. Review misses such as paraphrases and accidental matches. Record malformed or truncated responses as failures and undefined correlations as `NaN`.
- Use spaCy to mark names and check a sample by hand. Use VADER on the edit with cutoffs of `-0.05` and `0.05`, and record grammatical negation separately. For position, divide the edit’s midpoint by the token count of the two input sentences, use three equal bins, and keep the sentence role.
- For rarity, average lowercase target-word IDF using training pairs, with `log((N + 1) / (df + 1)) + 1` and `df = 0` for unseen words. Fix three rarity groups per edit type before seeing outcomes.
- Compare group scores using 2,000 bootstrap draws of whole examples, keeping their answer orders together. Report 95% intervals and raw and Benjamini–Hochberg-adjusted p-values from centered bootstrap contrasts. Show group counts and flag estimates that can’t be made. The remaining checks are secondary. One possible follow-up is whether longer edit spans are harder for the matcher to recognize.

</details>

## AI assistance

ChatGPT helped draft and debug the notebook, explain the statistical choices, and draft the edit-audit judgments. The author ran the notebook on Kaggle and discussed how to interpret label changes and examples where the explanation changed while the label stayed stable. The audit CSV identifies the AI-drafted judgments.

The verification checked the frozen input hashes, all 7,653 job IDs and saved responses, and the statistics recomputed from those responses. The implementation uses the published phi-CCT definition, the official e-SNLI data, and Qwen's published chat template. [Phi-CCT paper](https://arxiv.org/abs/2503.13445) · [e-SNLI data](https://github.com/OanaMariaCamburu/e-SNLI) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
