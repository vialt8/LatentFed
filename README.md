# LatentFed — standalone Kaggle notebooks

Implementations for **LatentFed: Reliability-Aware Latent Representations for Clustered Federated Learning under Statistical Heterogeneity**.

Every notebook contains its complete Python implementation: data loading, partitions, models, local training, server operations, evaluation, checkpoints and result export. Each file runs independently on Kaggle. No companion Python package, repository clone, account token or cloud-storage mount is required.

## Notebooks

| Notebook | Implementation |
| --- | --- |
| [01 — LatentFed full](01_latentfed_full.ipynb) | Convolutional autoencoder, MDFE, CALD, latent summaries, clustering, routing and classifier aggregation |
| [02 — LatentFed ablations](02_latentfed_ablations.ipynb) | Full control, w/o MDFE, w/o CALD and fully connected AE only |
| [03 — FedAC](03_fedac.ipynb) | Local regularization, PCA/cosine assignments and adaptive cluster-number updates |
| [04 — FedProto](04_fedproto.ipynb) | Local feature models, class-prototype regularization and server prototype averaging |
| [05 — IFCA](05_ifca.ipynb) | Candidate-model loss assignment and within-cluster model aggregation |
| [06 — HypCluster](06_hypcluster.ipynb) | Hypothesis selection by local training loss and hypothesis updates |
| [07 — ICFL](07_icfl.ipynb) | Model-similarity graphs, spectral partitioning and incremental cluster aggregation |
| [08 — WeCFL](08_wecfl.ipynb) | Model-space assignment, persistent client model bank and weighted cluster centers |
| [09 — FeSEM](09_fesem.ipynb) | Released first-K Lloyd initialization, center assignments and uniform center aggregation |
| [10 — CFL](10_cfl.ipynb) | Client-update geometry, minimax bipartition and the Algorithm 5 norm/gamma split conditions |

## Run on Kaggle

1. Import the desired `.ipynb` file into a fresh Kaggle notebook.
2. Enable **Internet** and select a **T4 GPU**.
3. Review **Select the experiment batch**, then choose **Run All**.
4. Download the ZIP linked by the final cell or from Kaggle Output.

The dependency cell installs the numerical packages and keeps Kaggle's compatible PyTorch/torchvision pair. If packages were imported before an update, restart the kernel once and choose Run All. Use one fresh kernel per notebook.

The method code is organized into named sections. Look for the model/objective and local-training sections, the method's clustering/aggregation sections, and **initialization and round update**. LatentFed additionally separates MDFE, the autoencoder, CALD, latent summaries, routing and prototypes.

## Experiment selection

All notebooks enable both datasets and the complete experimental grid:

```python
DATASETS = ["cifar10", "emnist"]
ALPHAS = [0.3, 0.5, 0.7]
SEEDS = [42, 101, 2024, 3407, 8888]
```

To run one dataset, concentration and seed at a time, select singleton lists:

```python
DATASETS = ["emnist"]
ALPHAS = [0.3]
SEEDS = [42]
```

Each selected real-data experiment still executes 100 rounds. A full method grid contains 30 experiments. The ablation notebook contains 120 experiments when all four variants are selected; set `VARIANTS` to the desired subset to run variants separately.

## Shared experimental protocol

| Setting | Value |
| --- | --- |
| Total clients | 20 |
| Participating clients in every round | **20 — full participation** |
| Communication rounds | 100 |
| Local epochs | 1 |
| Local minibatch size | 32 |
| Classifier | Three convolutional blocks and a 256-unit feature layer |
| Classifier optimizer | Adam |
| Classifier learning rate | CIFAR-10: `5e-4`; EMNIST: `1e-4` |
| Weight decay | `1e-4` |
| Client allocation | Seeded class-wise Dirichlet allocation |
| Validation holdout | Approximately 10% within each client's training classes |
| Final evaluation | Official test split |
| Seed endpoint | Mean of rounds 91–100 |

For a given dataset, concentration and seed, every notebook uses the same client sample indices, train/validation/test separation, initial CNN state, full-participation schedule and per-client minibatch/augmentation streams. Validation samples are excluded from local optimization. Loss-based assignments and latent summaries use the deterministic training view; test labels are used only for evaluation. Partition, initialization and participation hashes are recorded in each run manifest.

Method-specific objectives and server operations are explicit in their notebooks:

- **LatentFed:** latent dimension 128; autoencoder learning rate `5e-4`; MDFE and fixed-split CALD; 20-round CALD warm-up followed by a 20-round ramp; no AE pretraining rounds.
- **FedAC:** local coefficients `mu=0.01` and `lambda=0.01`, two initial clusters, PCA updates every ten rounds and adaptive cluster-number tuning.
- **FedProto:** prototype-loss coefficient `0.1` on CIFAR-10 and `1.0` on EMNIST. Predictions use the nearest class prototype in the learned feature space; models remain local.
- **IFCA / HypCluster:** two candidate models, local-loss assignment, within-cluster aggregation and reassignment after local updates.
- **ICFL:** clustering starts at round 10 and repeats every five rounds, with two to four clusters and the configured similarity/split thresholds.
- **WeCFL:** two centers, model-space assignments before and after local training, and aggregation of the persistent client model bank.
- **FeSEM:** three centers; cross-entropy local training; released uniform center aggregation and a 150-round assignment-refresh interval. The default 100-round run retains its initialized assignments.
- **CFL:** one initial root cluster; the first 20 rounds calibrate the norm thresholds within the 100-round budget. Subsequent splits apply the minimax partition and gamma condition. Children inherit the updated parent, then train separately in the next round.

IFCA, HypCluster and WeCFL use a client warm-start to initialize their cluster models. This initialization is recorded separately from the 100 federated rounds. The common classifier optimizer and learning rates apply to every baseline.

## Datasets

**CIFAR-10** downloads the original Python archive from the pinned Hugging Face mirror `MIT-OL-AI-D/cifar-10-python`, revision `a48007227f9e2cd0af96175f4afb5ac4965e261b`. Archive MD5: `c58f30108f718f92721af3b95e74349a`. Archive/member integrity checks retain official sample ordering.

**EMNIST** uses torchvision's **Balanced** split with 47 classes. Train and test images receive the orientation correction and normalization. Downloads are cached under the notebook's output directory. No Kaggle Add Input dataset or Hugging Face credentials are needed.

## LatentFed ablations

| Variant | Autoencoder | MDFE | CALD |
| --- | --- | --- | --- |
| `full` | Convolutional | Enabled | Fixed 64/64 latent split |
| `no_mdfe` | Convolutional | Identity bypass | Enabled |
| `no_cald` | Convolutional | Enabled | Disabled |
| `ae_only` | Fully connected | Absent | Disabled |

AE only uses `Flatten → Linear(input,512) → ReLU → Linear(512,128)` and the decoder `Linear(128,512) → ReLU → Linear(512,input) → Reshape`, with a linear reconstruction output. Input dimensions are 3072 for CIFAR-10 and 784 for EMNIST.

All variants retain classifier cross-entropy and the base AE objective: reconstruction MSE with weight `1.0` plus the class-structured term with weight `0.05`. Disabling CALD zeros its invariance, clustering, margin and routing terms. The federated classifier/clustering pipeline remains active.

## Outputs, summaries and resume

Each notebook writes to its own directory under `/kaggle/working`. Run folders are separated by method, variant, dataset, concentration and seed. They contain:

- `config.json` and `manifest.json`: settings, runtime identity, source-code identity and shared-protocol fingerprints;
- `round_metrics.csv` and `client_metrics.csv`: recorded trajectories and client-level test counts;
- `round_diagnostics.jsonl`: assignments and method-specific diagnostics;
- `checkpoint.pt`: model/server state and random streams;
- `seed_summary.json`: the completed seed endpoint;
- `threshold_calibration.json` for CFL.

The method output directory also contains `five_seed_summary.csv`, `seed_coverage.csv` and `results.zip`. Accuracy values in round/client logs are fractions. Five-seed accuracy summaries are percentages and report the mean and **sample standard deviation (`ddof=1`)** across the five seed endpoints. Seed endpoints average rounds 91–100.

W-Acc pools correct/total test counts. M-Acc averages the 20 client accuracies. MC, BC and WC are the mean, best and worst pooled cluster accuracies. FedProto does not assign client clusters, so its cluster metrics are not applicable. Geometric diagnostics use the representation defined by each method.

`RESUME = True` restores matching local checkpoints and reuses completed runs. To continue in another Kaggle session, set `INCLUDE_CHECKPOINTS_IN_ZIP = True`, download the ZIP and restore the same output hierarchy before running again. Resume checks scientific settings, live method definitions, data fingerprints and runtime identity. Raw datasets are cached separately and excluded from result ZIPs.

## Short execution check

Set `SMOKE_TEST = True` to run synthetic images for both dataset shapes. This check retains 20/20 clients, uses two rounds and shortens the CALD/ICFL/CFL diagnostic schedules. Synthetic outputs are stored under `smoke` and excluded from real-data seed aggregation. Set it to `False` for the documented experimental grid.

## GitHub upload and licenses

Upload the **ten notebooks**, this README and [LICENSE](LICENSE) to the repository root. Each notebook also works as an individual Kaggle upload.

The project code is distributed under Apache-2.0. The incorporated FeSEM routines retain the BSD 2-Clause notice embedded in the FeSEM notebook. Their source is the released [multi-center-fed-learning implementation](https://github.com/mingxuts/multi-center-fed-learning/tree/e2066819f9e598a5e35b7fb69354ab475ee07b91). CFL is the paper-based implementation of Algorithms 1 and 5. Original revision-notebook fingerprints are retained in notebook metadata.
