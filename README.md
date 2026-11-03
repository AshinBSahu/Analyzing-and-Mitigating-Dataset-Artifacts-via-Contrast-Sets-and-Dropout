# Analyzing and Mitigating Dataset Artifacts via Contrast Sets and Dropout

Trains an ELECTRA-small NLI model on SNLI, evaluates it on generated contrast sets, fine-tunes it on a mix of SNLI and contrast examples, and analyzes the results.

## Setup

Requires Python 3. Run everything from the repository root.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

SNLI, the base model (`google/electra-small-discriminator`) and the grammar model (`textattack/roberta-base-CoLA`) download automatically into `~/.cache/huggingface` (set `HF_HOME` to change this). NLTK data is downloaded by `generate_contrast_set.py`. The device is picked automatically (Apple MPS, then CUDA, then CPU).

## Full reproduction order

To reproduce everything from scratch, run the steps below in this order:

1. Generate the contrast sets (optional — already included in the repo).
2. Train the baseline ELECTRA-small model on SNLI.
3. Evaluate the baseline model.
4. Analyze the baseline model.
5. Fine-tune the baseline on SNLI plus the contrast set.
6. Evaluate the fine-tuned model.
7. Analyze the fine-tuned model.
8. Run the grammar analysis.

The base model (`google/electra-small-discriminator`) downloads automatically, but it is
only pretrained, not an NLI model. Steps 2 and 5 train it into the `model-baseline` and
`model-finetune` models that every evaluation and analysis step loads. These trained
models are not stored in the repository, so run the training steps before any evaluation
or analysis step. Each command is given in full in the sections below.

## Run

Trained models are not in the repository, so the two training steps are required before any evaluation or analysis step. The generated datasets in `synthesized_data/` are already included, so step 1 is optional.

### 1. Generate the contrast sets (optional)

```bash
python ./generate_contrast_set.py validation
python ./generate_contrast_set.py test
```

This writes new timestamped files to `synthesized_data/` that may differ from the ones already there. The commands below use the files already in the repo (`..._20251107065003.jsonl` for validation, `..._20251107065322.jsonl` for test). If you generate new ones, replace those file names.

### 2. Train the baseline model

```bash
python run.py --do_train --task nli --dataset snli --output_dir ./trained_model/model-baseline/
```

This uses the defaults (3 epochs, batch size 8, about 206,000 steps), which takes many hours on a laptop. A checkpoint (about 150 MB) is written every 500 steps. Add `--save_total_limit 1` to keep only the latest one.

### 3. Evaluate the baseline model

```bash
python run.py --do_eval --task nli --dataset ./synthesized_data/snli_original_validation_9842_20251107065003.jsonl --model ./trained_model/model-baseline/ --output_dir ./eval_output/
python run.py --do_eval --task nli --dataset ./synthesized_data/snli_contrast_validation_9842_20251107065003.jsonl --model ./trained_model/model-baseline/ --output_dir ./eval_output/
python run.py --do_eval --task nli --dataset ./synthesized_data/snli_original_test_9824_20251107065322.jsonl --model ./trained_model/model-baseline/ --output_dir ./eval_output/
python run.py --do_eval --task nli --dataset ./synthesized_data/snli_contrast_test_9824_20251107065322.jsonl --model ./trained_model/model-baseline/ --output_dir ./eval_output/
```

### 4. Analyze the baseline model

```bash
python ./dataset_analysis.py ./trained_model/model-baseline/ ./synthesized_data/snli_original_validation_9842_20251107065003.jsonl ./synthesized_data/snli_contrast_validation_9842_20251107065003.jsonl
python ./dataset_analysis.py ./trained_model/model-baseline/ ./synthesized_data/snli_original_test_9824_20251107065322.jsonl ./synthesized_data/snli_contrast_test_9824_20251107065322.jsonl
```

### 5. Fine-tune on SNLI plus the contrast set

```bash
python run.py --do_train --task nli --dataset snli --fine_tune yes --contrast_dataset ./synthesized_data/snli_contrast_validation_9842_20251107065003.jsonl --model ./trained_model/model-baseline/ --output_dir ./trained_model/model-finetune/ --max_steps 12500
```

This mixes a random 15% sample of SNLI train with the validation contrast set. The sample is not seeded, so results vary slightly between runs.

### 6. Evaluate the fine-tuned model

```bash
python run.py --do_eval --task nli --dataset ./synthesized_data/snli_original_validation_9842_20251107065003.jsonl --model ./trained_model/model-finetune/ --output_dir ./eval_output/
python run.py --do_eval --task nli --dataset ./synthesized_data/snli_contrast_validation_9842_20251107065003.jsonl --model ./trained_model/model-finetune/ --output_dir ./eval_output/
python run.py --do_eval --task nli --dataset ./synthesized_data/snli_original_test_9824_20251107065322.jsonl --model ./trained_model/model-finetune/ --output_dir ./eval_output/
python run.py --do_eval --task nli --dataset ./synthesized_data/snli_contrast_test_9824_20251107065322.jsonl --model ./trained_model/model-finetune/ --output_dir ./eval_output/
```

### 7. Analyze the fine-tuned model

```bash
python ./dataset_analysis.py ./trained_model/model-finetune/ ./synthesized_data/snli_original_validation_9842_20251107065003.jsonl ./synthesized_data/snli_contrast_validation_9842_20251107065003.jsonl
python ./dataset_analysis.py ./trained_model/model-finetune/ ./synthesized_data/snli_original_test_9824_20251107065322.jsonl ./synthesized_data/snli_contrast_test_9824_20251107065322.jsonl
```

### 8. Grammar analysis of the contrast set

```bash
python analyze_grammar_contrast_set.py
```

This reads `data_outputs/model-baseline_validation-dataset_contrast_cartography_results.csv`, which is created by the first `dataset_analysis.py` run in step 4.

## Notes

- Always pass `--dataset`. `run.py` needs it, and the training commands use `--dataset snli`.
- When using `--fine_tune yes`, always pass `--contrast_dataset`. Its default path points to a folder that does not exist.
- Keep the folder names `model-baseline` and `model-finetune`. `dataset_analysis.py` builds its output file names from them.
- `eval_output/` keeps only the most recent `--do_eval` run. The accuracy is printed to the console.
- The training commands above differ from the original project notes only in the output folder names. The notes trained into `./trained_model/` and then renamed the results.
