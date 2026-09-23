# Model Card

## Model summary

This repository provides a compact public reference implementation of a Hybrid Bayesian LSTM for bodily emotion recognition from low-level pose dynamics and higher-level Laban / kinematic descriptors.

The model combines:

- a bidirectional LSTM for pose trajectories;
- a bidirectional LSTM for higher-level motion descriptors;
- feature fusion;
- a variational Bayesian prediction head;
- Monte Carlo sampling for uncertainty estimation.

## Intended use

This implementation is intended for:

- research and education;
- temporal representation-learning experiments;
- uncertainty-aware classification prototypes;
- robustness testing under noisy pose observations;
- studying fusion of low-level and interpretable high-level motion features.

It is not intended for clinical diagnosis, surveillance, or other high-stakes decisions.

## Inputs

Reference input shapes:

- pose sequence: `[B, T, J, C]`
- LMA / high-level feature sequence: `[B, T, 33]`

The public implementation defaults to 17 joints and 3-D coordinates, but these dimensions are configurable.

## Outputs

The reference model produces:

- 7-way emotion logits;
- a fused temporal representation;
- a variational KL term for Bayesian regularisation.

Repeated stochastic forward passes can also produce:

- mean class probabilities;
- predictive entropy;
- mean class variance.

## Robustness

The repository contains generic utilities for:

- Gaussian-noise perturbation;
- FGSM pose perturbation.

These are included to demonstrate robustness-evaluation interfaces. They are not substitutes for the full evaluation protocol described in the paper.

## Privacy and ethical considerations

Pose and motion representations reduce reliance on raw appearance, but they are not inherently anonymous. Movement patterns can still reveal sensitive or identifying information.

Any real deployment should use appropriate consent, access control, retention limits, and data minimisation.

## Limitations

- Synthetic inputs are used in public smoke tests.
- Restricted datasets and paper checkpoints are not distributed.
- The code is a compact public reference implementation, not the exact internal training codebase.
- Affect labels are context-dependent and should not be interpreted as direct measurements of a person's internal mental state.
- Predictive uncertainty from a Bayesian approximation does not guarantee calibrated real-world uncertainty.

## Out-of-scope use

This repository should not be used to:

- infer sensitive traits about individuals;
- make employment, medical, legal, insurance, or other consequential decisions;
- identify people from motion;
- claim ground-truth emotional state from movement alone.

## Associated publication

**Shuang Wu and Daniela M. Romano**  
*Robust emotion recognition using hybrid Bayesian LSTM based on Laban movement analysis*  
AI Open, 2025  
DOI: 10.1016/j.aiopen.2025.09.002
