# Transformer Markov Scaling Experiments

Research experiments on decoder-only Transformers trained on synthetic binary Markov processes.

This repository investigates whether the high-error regime observed for higher-order Markov processes can be reduced by increasing the available training data or by increasing generic model/training capacity.

It also contains an auxiliary context-corruption experiment examining how bit-flip noise affects statistical fidelity.

## Main Experiments

### 1. Data Scaling

- `notebooks/data_scaling.ipynb`  
  Trains the baseline Transformer architecture while increasing the training-sequence length / data budget for higher Markov orders.

- `notebooks/data_scaling_evaluation.ipynb`  
  Evaluates the resulting models using transition-kernel recovery metrics such as Kullback-Leibler divergence (KLD) and mean absolute error (MAE).

This experiment tests whether increasing the available training data alone removes the high-error regime.

### 2. Model / Compute Scaling

- `notebooks/model_scaling.ipynb`  
  Compares baseline, medium, and larger Transformer configurations by increasing depth, embedding dimension, attention-head count, and training iterations.

- `notebooks/model_scaling_evaluation.ipynb`  
  Evaluates the scaling runs using the same transition-kernel recovery metrics.

This should be interpreted as a generic model/compute scaling experiment rather than a perfectly isolated architecture ablation, because the larger configurations also use larger training budgets.

## Auxiliary Experiment: Context Corruption

- `notebooks/auxiliary/context_corruption.ipynb`

This experiment studies how bit-flip corruption in the input context affects the statistical structure of Transformer-generated sequences.

Two evaluation paths are compared:

1. **Direct k-gram estimation**  
   A binary Markov sequence is corrupted and the transition kernel is estimated directly from the noisy sequence.

2. **Transformer generation**  
   The input context is corrupted, the trained Transformer generates a continuation, and the transition kernel is estimated from the generated sequence.

The resulting transition statistics are compared with the known source process using KLD and MAE.

In the tested setup, direct k-gram estimates degrade as bit-flip corruption increases, while the Transformer-generated transition statistics are comparatively insensitive to the tested context corruption.

This should not be interpreted as evidence that the Transformer ignores its context. A safer interpretation is that the learned transition structure dominated the effect of the tested input corruption in this experiment.

## Results

Saved evaluation outputs for the scaling experiments are available under:

```text
results/
├── data_scaling/
│   └── evaluation_results.json
└── model_scaling/
    └── evaluation_results.json
```

Selected figures are stored in `plots/`.

## Main Findings

The experiments support the narrower conclusions that:

- increasing training data alone did not remove the high-error regime;
- increasing generic model/compute capacity did not qualitatively remove it;
- in the auxiliary corruption experiment, Transformer-generated transition statistics were comparatively robust to the tested context bit flips.

These experiments are not intended as a formal scaling-law study.

## Setup

Install the main dependencies with:

```bash
pip install torch numpy matplotlib scipy
```

## Repository Structure

```text
transformer-markov-scaling/
├── notebooks/
│   ├── data_scaling.ipynb
│   ├── data_scaling_evaluation.ipynb
│   ├── model_scaling.ipynb
│   ├── model_scaling_evaluation.ipynb
│   └── auxiliary/
│       └── context_corruption.ipynb
├── results/
│   ├── data_scaling/
│   └── model_scaling/
├── plots/
├── README.md
├── requirements.txt
└── .gitignore
```

## Notes

Model checkpoints are intentionally excluded because of their size.

Notebook outputs are preserved where available so that the reported results can be inspected directly without rerunning every experiment.

The context-corruption notebook is based on a local adaptation of the Markov-Transformer framework used for the original project. Baseline utilities are retained inside the notebook so the experiment remains self-contained.
