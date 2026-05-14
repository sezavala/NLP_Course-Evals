# Experiment Reproduction Guide

This file is a compact runbook for reproducing the NLP course-evaluation experiments. It focuses on setup, run order, inputs, outputs, and the few settings needed to compare results.

## Environment

Use Python 3.10 or newer from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Transformer models download from Hugging Face on first use. The Ollama-backed models require a local Ollama server:

```bash
ollama pull llama3
ollama pull gemma3:latest
ollama serve
```

Keep `ollama serve` running in another terminal before running Llama3 or Gemma experiments.

## Inputs

The experiments use the shared topic definitions, rubrics, and sample feedback in `data.py`.

The comparison scripts use two manual baselines:

- `HUMAN_CATEGORIZED_OUTPUT.csv`: manual multi-label topic assignments. Expected first column is `Feedback`; remaining columns are topic names with any non-empty value marking an assigned topic.
- `HUMAN_SENTIMENT_BASELINE.csv`: manual topic-level sentiment scores. Expected columns are `Feedback`, `Topic`, `Sentiment`, `Score`, and `Reasoning`.

Keep topic names consistent with `TOPIC_DEFS` in `data.py`; mismatched topic columns will not compare cleanly.

## Recommended Run Order

Run commands from the repository root so imports resolve consistently.

### 1. Topic Classification

```bash
python3 -m experiments.LLama3.classification_model
python3 -m experiments.Gemma.classification_model
python3 -m experiments.roBERTa.classification_model
python3 -m experiments.DistilroBERTa.classification_model
```

Main outputs:

- `results/Llama3/LLAMA_OUTPUT.json`
- `results/Llama3/LLAMA_OUTPUT.csv`
- `results/Gemma/GEMMA_OUTPUT.json`
- `results/Gemma/GEMMA_OUTPUT.csv`
- `results/roBERTa/roberta_baseline_output.json`
- `results/roBERTa/roBERTa_OUTPUT.csv`
- `results/DistilroBERTa/DISTILROBERTA_OUTPUT.json`
- `results/DistilroBERTa/DISTILROBERTA_OUTPUT.csv`

Important settings:

- Llama3 uses Ollama model `llama3`.
- Gemma uses Ollama model `gemma3:latest`.
- roBERTa uses `roberta-large-mnli`.
- DistilroBERTa uses `cross-encoder/nli-distilroberta-base`.
- Transformer classification assigns topics using a 0.50 confidence threshold.

### 2. Topic Model Comparison

```bash
python3 -m comparison.model_comparison
```

Outputs:

- `results/topic_model_comparison.csv`
- `results/topic_model_per_topic_metrics.csv`
- `results/topic_model_unmatched.csv`
- `results/visualizations/classification_model_comparison_poster.png`

The script reports matched comments, exact topic-set match rate, precision, recall, micro F1, and macro F1 against `HUMAN_CATEGORIZED_OUTPUT.csv`.

### 3. Sentiment Analysis

All sentiment models intentionally use `results/Llama3/LLAMA_OUTPUT.json` as the shared topic-classification input. Run the Llama3 classifier first.

```bash
python3 -m experiments.LLama3.sentiment_model
python3 -m experiments.Gemma.sentiment_model
python3 -m experiments.roBERTa.sentiment_model
python3 -m experiments.DistilroBERTa.sentiment_model
```

Outputs:

- `results/Llama3/LLAMA_SENTIMENT.json`
- `results/Llama3/LLAMA_SENTIMENT.csv`
- `results/Gemma/GEMMA_SENTIMENT.json`
- `results/Gemma/GEMMA_SENTIMENT.csv`
- `results/roBERTa/ROBERTA_SENTIMENT.json`
- `results/roBERTa/ROBERTA_SENTIMENT.csv`
- `results/DistilroBERTa/DISTILROBERTA_SENTIMENT.json`
- `results/DistilroBERTa/DISTILROBERTA_SENTIMENT.csv`

### 4. Sentiment Model Comparison

```bash
python3 -m comparison.compare_sentiment_models
```

Outputs:

- `results/sentiment_model_comparison.csv`
- `results/sentiment_model_metrics.json`

### 5. Visualizations

```bash
python3 -m comparison.create_example_visual
python3 -m comparison.visualize_comparison
```

Outputs are written to `results/visualizations/`.

## Main Pipeline

`main.py` is the combined Llama3 pipeline. It classifies comments, scores each assigned topic, summarizes each topic, and writes a combined report.

```bash
python3 main.py
```

Outputs:

- `results/combined/<COURSE_ID>_COMBINED_REPORT.json`
- `results/combined/<COURSE_ID>_COMBINED_REPORT.csv`
- `results/<COURSE_ID>.json`

The example input is currently embedded in `main.py` under `json_input`. For a new course, update `course_id` and `raw_comments`, or call `analysis_pipeline(course_id, raw_comments)` from another Python file.

## Reproducibility Notes

- LLM outputs can vary slightly between runs, even at low temperature.
- The Ollama experiments depend on the locally installed model versions.
- The transformer experiments depend on downloaded Hugging Face model revisions.
- No global random seed is set in the current scripts.
- Existing result files in `results/` may be overwritten by reruns.

## Troubleshooting

If Ollama calls fail, confirm the server is running:

```bash
curl http://localhost:11434/api/tags
```

If Python cannot import `experiments` or `comparison`, rerun the command from the repository root using `python3 -m ...` as shown above.

If transformer model downloads fail, check network access and rerun the same command after the download issue is resolved.
