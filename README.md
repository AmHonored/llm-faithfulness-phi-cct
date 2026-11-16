# Do Qwen's explanations reflect what changes its answers?

I gave Qwen a premise and a hypothesis from e-SNLI and asked it to explain its answer before choosing a label. Then I changed a small detail and asked again. Phi-CCT measures how closely mentions of the detail track changes in the answer.

**A small edit can change the explanation while the answer stays the same.** When an edit added *sad* to a premise about a girl pouring water, Qwen mentioned her mood and still predicted entailment. I wanted to know whether **rare inserted words** would change how closely those mentions track label flips. The [qualitative examples](results/test/qualitative_examples.csv) include this case and five others, with job IDs for looking up the raw responses.

## What I ran

I froze **1,000 distinct e-SNLI test pairs** with seed `42` and used a separate set of **10 validation pairs** for the smoke test. The edit rules tried an insertion and a deletion for each test pair. After 15 audit exclusions, 632 pairs had two eligible edits, 287 had one, and 81 had none. The [coverage table](results/test/coverage.csv) includes all 1,000 pairs, including cases where an edit or a response failed.

I reviewed the 20 sampled edits on **12 November 2026**, before the main run. The [audit CSV](data/prepared/edit_audit.csv) contains the decisions used for the run.

I ran **Qwen3-8B** on two Kaggle Tesla T4 GPUs, using FP16 without quantization, greedy decoding, and Qwen's chat template with thinking disabled. Qwen explained its choice before giving the answer letter. Each input ran in three balanced answer orders. I compared original and edited answers under the same order after mapping letters back to entailment, neutral, or contradiction.

## What I found

The full run completed all **7,653 prompts** in about **1 hour 48 minutes** of generation time. Of those, 25 responses hit the token limit and were excluded from scoring. Baseline accuracy was **68.2%** among parsed answers (2,039/2,989) and **68.0%** across all 3,000 baseline prompts when truncated responses count as incorrect.

| Edit | Valid comparisons | Phi-CCT | 95% bootstrap interval | Mention rate | Label flip rate |
|---|---:|---:|---:|---:|---:|
| Insertion | 2,402 | 0.237 | 0.185 to 0.287 | 55.1% | 18.8% |
| Deletion | 2,230 | 0.075 | 0.018 to 0.130 | 67.0% | 14.2% |

**Rare insertions showed no clear difference.** The rare-minus-common difference was −0.020, with a 95% interval from −0.142 to 0.093. Target position was the only comparison that showed a difference after adjustment (insertion p = 0.020, deletion p = 0.009). All early edits were in the premise and most late edits were in the hypothesis, so sentence role could help explain the pattern.

I also checked named entities, sentiment polarity, and answer-option position. Only 16 deletion examples involved named entities, so the deletion comparison was too small to test. The [two-page report](esnli_phi_cct_report.pdf), [plots](results/test/group_phi_cct.png), and [statistical results](results/test/variable_tests.csv) cover all five variables.

For insertion, I checked whether the edited explanation mentioned the new detail. For deletion, I checked whether the original explanation mentioned the removed detail. The rule matches the full sequence of target words, ignoring case. For example, it misses *the park* when the deleted phrase is *at the park*. A label flip measures how the model responded to the edit. The edited pairs have no gold labels for checking whether the new answer was correct.

## Reproduce the run

1. Import [esnli_phi_cct.ipynb](esnli_phi_cct.ipynb) into Kaggle. Turn **Internet** on and select **GPU T4 x2**.
2. Use **Restart & Run All** for the full experiment. The notebook prepares the pinned dataset, runs the smoke test, and evaluates the frozen test set. Set `RUN_FULL_EVALUATION = False` to stop after the smoke test. The original smoke test generated 78 responses in 63 seconds and projected roughly 1 hour 43 minutes for the main run, plus setup.
3. Download `results/test_results.zip` when the notebook finishes. To resume an interrupted run in a new session, restore `results/test/checkpoint.sqlite3` at the same path and use the same settings.

The [data manifest](data/prepared/manifest.json) records the dataset revision, sample hashes, and data preparation versions. The [run summary](results/test/summary.json) records the exact model revision, precision, hardware, memory use, and package versions. The notebook explains how I calculated training-only IDF, used spaCy and VADER to annotate edits, and drew 2,000 bootstrap samples while keeping all answer orders from the same source together.

## AI assistance

I used ChatGPT to help write and debug the notebook, drafting my audit and helped on working through the statistics. I ran the experiment on Kaggle and focused on examples where the explanation changed while the label stayed the same.

To check the results, I used a separate analysis that rebuilt all 7,653 prompts, reparsed the saved responses, and recomputed the scores, uncertainty intervals, and p-values. The implementation uses the metric from the [phi-CCT paper](https://arxiv.org/abs/2503.13445), with the [reference implementation](https://github.com/google-deepmind/corr_faith), [official e-SNLI data](https://github.com/OanaMariaCamburu/e-SNLI), and [Qwen's published template](https://huggingface.co/Qwen/Qwen3-8B) as references.
