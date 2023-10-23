# Evolving Connectivity for Recurrent Spiking Neural Networks Repository

This repository contains the implementation of the paper [Evolving Connectivity for Recurrent Spiking Neural Networks](https://arxiv.org/abs/2305.17650). It includes the Evolutionary Connectivity (EC) algorithm, Recurrent Spiking Neural Networks (RSNN), and the Evolution Strategies (ES) baseline implemented in JAX.

## Getting Started

### Prerequisites

1. [Install JAX](https://github.com/google/jax#installation)

2. [Install W&B](https://github.com/wandb/wandb) and log in to your account to view metrics

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Precautions

- Brax v1 is required (`brax<0.9`) to reproduce our experiments. Brax v2 has completely rewritten the physics engine and adopted a different reward function.
- Due to the inherent numerical stochasticity in Brax's physics simulations, variations in results can occur even when using a fixed seed.

## Usage

### Training EC with RSNN

To set parameters, use the command-line format of [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#id15). For example:

```
python ec.py task=humanoid
```

### Running experiment sets

To reproduce the Brax locomotion experiments using EC-RSNN:

```
python exp_launcher.py include=conf_experiment/ec_brax.yaml
```

To reproduce the ES experiments:

- Deep RNN (GRU, LSTM)

```
python exp_launcher.py include=conf_experiment/rnn_brax.yaml
```

- Densely weighted RSNN

```
python exp_launcher.py include=conf_experiment/dense_snn_brax.yaml
```

**Note**: The experiment launcher will automatically allocate all idle GPUs on your machine and run experiments in parallel.

## Citation

```
@inproceedings{wang2023evolving,
    title={Evolving Connectivity for Recurrent Spiking Neural Networks},
    author={Wang, Guan and Sun, Yuhao and Cheng, Sijie and Song, Sen},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=30o4ARmfC3}
}
```

## License

This project is licensed under the Apache License 2.0.
