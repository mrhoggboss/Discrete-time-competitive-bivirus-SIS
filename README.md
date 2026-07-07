# Discrete-Time Networked Competitive Bivirus SIS — figure reproduction

Minimal code to reproduce the numerical-example figures in

> S. Gracy, Y. Xu, J. Liu, T. Başar, C. A. Uribe,
> *Networked Competitive Bivirus SIS Model — Analysis of the Discrete-Time Case.*

The model is the discrete-time competitive bivirus SIS system

  x<sup>ℓ</sup>(k+1) = x<sup>ℓ</sup>(k) + h[ (I − X<sup>1</sup> − X<sup>2</sup>) B<sup>ℓ</sup> − D<sup>ℓ</sup> ] x<sup>ℓ</sup>(k),  ℓ = 1, 2.

## Contents

| File | Paper figure |
|---|---|
| `bivirus.py` | model + plotting utilities |
| `experiment_Theorem2.ipynb` | Fig. 1 — disease-free equilibrium |
| `experiment_Theorem3.ipynb` | Fig. 2 — single-virus endemic equilibrium |
| `experiment_Theorem4.ipynb` | Figs. 3 & 4 — stable boundary / coexistence cases |
| `experiment_Theorem6.ipynb` | Fig. 5 — necessary condition |
| `experiment_Theorem7.ipynb` | Fig. 6 — continuum of coexistence equilibria |

## Requirements

Python 3 with `numpy`, `matplotlib`, and `networkx`.

## Reproducing the figures

Open a notebook and run all cells.

All randomness is seeded, so the figures are fully reproducible.
