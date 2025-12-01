# 🧪 Evaluation

This directory contains scripts for evaluating trained models and generating performance metrics.

## Key Scripts

| File | Description |
| :--- | :--- |
| **[`eval.py`](eval.py)** | **Model Evaluation**. Loads a trained model checkpoint and evaluates it on a test set. Reports Accuracy, F1-Score, and Kappa. |
| **[`moabb_eval.py`](moabb_eval.py)** | **MOABB Benchmark**. Integrates with the Mother of All BCI Benchmarks (MOABB) framework to compare performance against state-of-the-art baselines. |

## Usage

To evaluate a saved model:
```bash
python eval/eval.py --model_path ../models_saved/best_model.pt --dataset bci42a
```
