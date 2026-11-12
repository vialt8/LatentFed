# LatentFed

Reproducible experiments for **LatentFed: Reliability-Aware Latent Representations for Clustered Federated Learning under Statistical Heterogeneity**. The repository provides the full method, module ablations, and two baselines under a common experimental protocol on **CIFAR-10 and EMNIST Balanced**.

| Notebook | Experiments |
| --- | --- |
| [01 — LatentFed full](notebooks/01_latentfed_full.ipynb) | Convolutional autoencoder, MDFE and CALD |
| [02 — LatentFed ablations](notebooks/02_latentfed_ablations.ipynb) | Full, w/o MDFE, w/o CALD, fully connected AE only |
| [03 — FedAC](notebooks/03_fedac.ipynb) | FedAC with matched partitions and training budget |

The notebooks contain setup, an explicit experiment grid, model inspection, execution, results and ZIP export. Each notebook enables `DATASETS = ["cifar10", "emnist"]`.

## Installation and first run

Use Python **3.12**. A CUDA GPU is the default for real-data experiments. Keep a compatible PyTorch/torchvision pair supplied by Kaggle, or install a pair using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Then, from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m jupyterlab
```

Open a notebook and choose **Run All**. Its configuration cell is the entry point for changing the dataset/alpha/seed batch. The scientific defaults are the same in all four notebooks.

For a CPU-only environment that runs the short validation checks:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m latentfed_repro.cli --method latentfed --smoke
```

The notebook setup installs the pinned numerical packages if needed. If it updates an already imported package, restart the kernel and choose Run All so every cell uses the installed version. PyTorch and torchvision are kept as the installed compatible pair; their versions and device are recorded in each manifest.

### Kaggle

Upload/open one of the four notebooks, enable **Internet** and a **T4 GPU**, and choose Run All. The setup cell locates the complete checkout or clones [vialt8/LatentFed](https://github.com/vialt8/LatentFed). Outputs are written under `/kaggle/working/latentfed_repro_outputs`. The final cell creates a downloadable ZIP, also available in Kaggle Output. Enable checkpoint inclusion before exporting if the run will continue in a later session.

## Experiment grid and budget

| Setting | Value |
| --- | --- |
| Datasets | CIFAR-10; EMNIST Balanced, 47 classes |
| Dirichlet concentration | 0.3, 0.5, 0.7 |
| Seeds | 42, 101, 2024, 3407, 8888 |
| Clients / participants each round | **20 / 20** |
| Communication rounds | **100** |
| Local epochs / batch size | **1 / 32** |
| Classifier optimizer | Adam, weight decay 1e-4 |
| Classifier learning rate | 5e-4 for CIFAR-10; 1e-4 for EMNIST |
| Autoencoder learning rate / latent dimension | 5e-4 / 128 |
| Validation holdout | Approximately 10% within each client's training classes |
| Evaluation | Official test split; validation samples are excluded from training |

One method/variant has **30 experiments**: 2 datasets × 3 concentrations × 5 seeds. The ablation notebook has **120 experiments**, including the full control. Running both LatentFed notebooks with the same output root reuses completed full-method runs.

To split the workload across sessions, select a real-data batch in the configuration cell, for example:

```python
DATASETS = ["cifar10", "emnist"]
ALPHAS = [0.3]
SEEDS = [42]
```

Each selected experiment still uses the full 100-round budget. Repeat for the remaining official seeds and concentrations, keeping the same configuration across methods. Completed seeds accumulate in the output tree; coverage and five-seed statistics are rebuilt automatically.

## Full method and ablations

| Variant key | Autoencoder | MDFE | CALD |
| --- | --- | --- | --- |
| `full` | Convolutional | Enabled | Fixed 64/64 latent split |
| `no_mdfe` | Convolutional | Identity bypass | Enabled |
| `no_cald` | Convolutional | Enabled | Disabled |
| `ae_only` | **Fully connected** | Absent | Disabled |

The fully connected [`FullyConnectedAutoencoder`](src/latentfed_repro/models.py) has encoder `Flatten → Linear(input,512) → ReLU → Linear(512,128)` and decoder `Linear(128,512) → ReLU → Linear(512,input) → Reshape`. The input dimension is 3072 for CIFAR-10 and 784 for EMNIST. Its linear reconstruction output accommodates normalized images.

All variants retain classifier cross-entropy and the base autoencoder objective: reconstruction MSE (weight 1.0) plus the class-structured term (weight 0.05). Disabling CALD zeros its invariance, clustering, margin and routing terms. Disabling MDFE bypasses the channel gate. The same server clustering and classifier aggregation pipeline is used across the four variants. See the complete [protocol](docs/EXPERIMENT_PROTOCOL.md).

## Data and matched baselines

CIFAR-10 downloads the original Python archive from a [pinned Hugging Face mirror](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/tree/a48007227f9e2cd0af96175f4afb5ac4965e261b), verifies its archive and member checksums, and preserves the official sample ordering. EMNIST uses torchvision's **Balanced** train/test splits and applies the orientation correction before normalization. The first run needs Internet; subsequent runs reuse the local data cache. Kaggle Add Input and Hugging Face credentials are unnecessary.

FedAC and FedProto use the **same sample-index partitions, validation holdout, classifier initial state, minibatch/augmentation streams and participation schedule** as LatentFed for a given dataset, concentration and seed. Run manifests record hashes for the partitions, initialization and schedule. Method-specific local and server operations are preserved from the supplied benchmark implementations. FedAC predicts with client classifiers; FedProto's primary prediction uses nearest global class prototypes, with local-head accuracy recorded separately.

The released baselines define a common-protocol comparison. Earlier baseline result archives used independently constructed train/test partitions; their numerical rows belong to that earlier partition protocol. Newly generated results are identified by `matched_latentfed_v1` and their configuration/source hashes.

## Results, aggregation and resuming

Each run writes `config.json`, `manifest.json`, `round_metrics.csv`, `client_metrics.csv`, `round_diagnostics.jsonl` and a resumable `checkpoint.pt`. Completed 100-round real-data runs also write `seed_summary.json`.

An endpoint averages **rounds 91–100 within one seed**. `five_seed_summary.csv` then reports the mean and **sample standard deviation (ddof=1) across the five seed endpoints**. Accuracy columns in that file are percentages. `seed_coverage.csv` reports completed real-data seed coverage. A publication row requires all five official seeds with matching settings and source. Undefined clustering metrics remain missing. FedProto has no client clusters, so MC/BC/WC and geometric cluster scores are not applicable. Round metrics retain fractional accuracies.

With `RESUME = True`, a run resumes its local checkpoint or reuses its completed result. Source, scientific settings, data/initialization/schedule hashes and runtime identity are checked. Restore the run folder hierarchy when moving an output archive to another session. Checkpoints are excluded from the compact results ZIP by default; set `INCLUDE_CHECKPOINTS_IN_ZIP = True` to include them. Raw datasets are cached separately and are never included in result ZIPs.
