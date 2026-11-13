# Do Qwen’s explanations reflect what changes its answers?

We’re testing Qwen on e-SNLI, where it reads two sentences, explains how they relate, and then picks a label. We make a small edit and check two things. Did Qwen change its label? Did its explanation mention the edited detail? **Phi-CCT** measures how closely those two things line up across examples.

Our main question is whether the score differs when we insert a common word or a rare one. A rare word might stand out and get mentioned more often.

## How we’ll run it

1. Pick 1,000 distinct examples from the e-SNLI test split using seed `42`. Save the sample IDs, label counts, and planned edits before looking at any model output.
2. Try to add one detail and remove one detail per example. Review at least 20 edits across both types. Check whether each edit reads naturally, makes sense, and might change the correct label. Record rejected edits and any examples where we fall short.
3. Try the notebook on 10 separate validation examples in Kaggle first. Use that trial to check memory, estimate runtime, and test that the notebook can resume.
4. Run `Qwen/Qwen3-8B` on the full sample in a standalone Python/PyTorch notebook. It writes the explanation first and then gives the label. The run uses greedy decoding, thinking mode off, and no quantization.

## Run the data preparation

Import `esnli_phi_cct.ipynb` into Kaggle, enable internet, leave the accelerator off, and run the data preparation cells. The notebook downloads the pinned e-SNLI revision, verifies that the test and smoke samples match the original run, and saves the outputs in `data/prepared/`. In `data/prepared/edit_audit.csv`, the 20 rows marked `planned_sample` are the edits selected for review before running the model.

The revised rules proposed both edits for 640 of the 1,000 examples. The audit records the 20 planned reviews and a follow-up check for incomplete sentences after deletion. It flags 15 proposals to exclude before inference, leaving 632 examples with two eligible edits, 287 with one, and 81 with zero. The audit identifies which decisions were drafted with AI assistance.

## Run the validation check

Import the notebook into Kaggle, select **GPU T4 x2**, turn **Internet** on, and choose **Run All**. The notebook runs 78 prompts from 10 validation examples across the three answer orders. It uses FP16 weights without quantization, greedy decoding, and Qwen's chat template with thinking disabled.

The results are in `results/smoke/greedy/`. `responses.csv` contains the raw responses and parsed labels. `summary.json` records the model revision, hardware, response statuses, and estimated runtime. Keep `checkpoint.sqlite3` to resume with the same inputs and settings.

The validation run parsed all 78 responses and reused the first 8 after reopening its checkpoint. Qwen answered 24 of 30 baseline prompts correctly across the three answer orders. Generation took 63 seconds, which projects to about 1 hour 43 minutes for the 7,653 planned test prompts after audit exclusions, plus setup time. These timings came from Kaggle T4 x2.

## What we'll compare

| Check | What we’ll look at |
|---|---|
| Named entity | Does the edit touch a name, such as a person or place? |
| Polarity | Is the edited detail positive, negative, or neutral in sentiment? |
| Prompt position | Does the edit appear early, in the middle, or late in the input? |
| Word rarity | How common are the edited words in the training data? |
| Answer order | Does the score change when the original correct label appears as A, B, or C? |

We’ll run all three answer orders on the same examples and match each edit to its original answer under the same order. For an insertion, we look for the new detail in the edited explanation. For a deletion, we look for the removed detail in the original explanation. We’ll report the two edit types separately.

<details>
<summary>Scoring and analysis details</summary>

- Pin the dataset revision and record the original text, edited text, and exact span for each edit. Run each prompt in a fresh context. Reuse the baseline for both edits within each answer order, for up to 9,000 generations in total.
- `I_D` marks a change in the meaning of the predicted label after decoding its letter. `E_D` marks a case-insensitive whole-word or stem match to the edit. Review misses such as paraphrases and accidental matches. Record malformed or truncated responses as failures and undefined correlations as `NaN`.
- Use spaCy to mark names and check a sample by hand. Use VADER on the edit with cutoffs of `-0.05` and `0.05`, and record grammatical negation separately. For position, divide the edit’s midpoint by the token count of the two input sentences, use three equal bins, and keep the sentence role.
- For rarity, average lowercase target-word IDF using training pairs, with `log((N + 1) / (df + 1)) + 1` and `df = 0` for unseen words. Fix three rarity groups per edit type before seeing outcomes.
- Compare group scores using 2,000 bootstrap draws of whole examples, keeping their answer orders together. Report 95% intervals and raw and Benjamini–Hochberg-adjusted p-values from centered bootstrap contrasts. Show group counts and flag estimates that can’t be made. The remaining checks are secondary. One possible follow-up is whether longer edit spans are harder for the matcher to recognize.

</details>

## What we’ll share

The notebook, CSVs, and short report will show coverage for all 1,000 examples, failed runs, baseline accuracy, counts for all four mention and label-change outcomes, and at least five examples you can trace back to the raw outputs. We’ll add the exact run steps, model revision, package versions, hardware, precision, runtime, and AI-use notes after the experiment.
