# Transformer Markov Scaling Experiments

Research experiments on decoder-only Transformers trained on synthetic binary Markov processes.

This repository contains two scaling studies used to investigate whether the high-error regime observed for higher-order Markov processes can be removed by increasing training data or by increasing generic model/training capacity.

## Experiments

### 1. Data scaling

- `notebooks/data_scaling.ipynb`  
  Trains the baseline Transformer architecture while increasing the available training-sequence length / data budget for higher Markov orders.

- `notebooks/data_scaling_evaluation.ipynb`  
  Evaluates the resulting models using transition-kernel recovery metrics such as KLD and MAE.

The purpose of this experiment is to test whether increasing the data budget alone removes the high-error regime.

### 2. Model / compute scaling

- `notebooks/model_scaling.ipynb`  
  Compares baseline, medium, and larger Transformer configurations by increasing depth, embedding dimension, attention-head count, and training iterations.

- `notebooks/model_scaling_evaluation.ipynb`  
  Evaluates the capacity-scaling runs using the same Markov-kernel recovery metrics.

This experiment should be interpreted as a generic model/compute scaling study rather than a perfectly isolated architecture ablation, because the larger configurations also use larger training budgets.

## Results

Saved evaluation outputs are available under:

```text
results/
├── data_scaling/
│   └── evaluation_results.json
└── model_scaling/
    └── evaluation_results.json
```

Selected figures are stored in `plots/`.

## Main conclusion

The experiments support the narrower conclusions that:

- increasing training data alone did not remove the high-error regime;
- increasing generic model/compute capacity did not qualitatively remove it either.

These experiments are not intended as a formal scaling-law study.

## Setup

Install the main dependencies with:

```bash
pip install torch numpy matplotlib scipy
```

## Repository structure

```text
transformer-markov-scaling/
├── notebooks/
│   ├── data_scaling.ipynb
│   ├── data_scaling_evaluation.ipynb
│   ├── model_scaling.ipynb
│   └── model_scaling_evaluation.ipynb
├── results/
│   ├── data_scaling/
│   └── model_scaling/
├── plots/
├── README.md
├── requirements.txt
└── .gitignore
```

## Notes

Model checkpoints are intentionally excluded because of their size. Notebook outputs are preserved where available so the reported results can be inspected directly without rerunning every experiment.
