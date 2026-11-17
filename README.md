# LatentFed — standalone Kaggle notebooks

Code for **LatentFed: Reliability-Aware Latent Representations for Clustered Federated Learning under Statistical Heterogeneity**.

Each notebook contains its complete implementation in ordinary Python cells. Upload the individual `.ipynb` file to Kaggle and choose **Run All**. The notebooks require no companion Python files, repository clone or Hugging Face token.

| Notebook | Included implementation |
| --- | --- |
| [LatentFed full](01_latentfed_full.ipynb) | CNN, convolutional autoencoder, MDFE, CALD, local training, latent clustering, prototypes and federated loop |
| [LatentFed ablations](02_latentfed_ablations.ipynb) | Full, w/o MDFE, w/o CALD and fully connected AE only |
| [FedAC baseline](03_fedac.ipynb) | Local regularization, PCA/cosine assignments, adaptive cluster count, aggregation and federated loop |

## Running on Kaggle

1. Import one of the three notebooks into Kaggle.
2. Enable **Internet** and a **T4 GPU** in notebook settings.
3. Review **Select the experiment batch**, then choose **Run All**.
4. Download the ZIP from the final cell or Kaggle Output.

The first cell installs the numerical dependencies. It keeps Kaggle's compatible PyTorch/torchvision pair. If a dependency was already imported before its version was updated, restart the kernel once and Run All. Use a fresh kernel for each notebook.

The code is organized into named sections: configuration; seeded streams; data; matched partitions/loaders; model definitions; losses and training; server operations; evaluation; checkpoints; the federated loop; results and download. In the full-method notebook, **MDFE gate**, **Convolutional autoencoder**, **Base latent loss and CALD**, **Local classifier and autoencoder training**, and **LatentFed federated loop** contain the method itself.

## Common experiment protocol

All three notebooks enable:

```python
DATASETS = ["cifar10", "emnist"]
ALPHAS = [0.3, 0.5, 0.7]
SEEDS = [42, 101, 2024, 3407, 8888]
```

| Setting | Value |
| --- | --- |
| Clients | 20 |
| Participating clients per round | **20 (100%)** |
| Communication rounds | 100 |
| Local epochs / batch size | 1 / 32 |
| Classifier optimizer / weight decay | Adam / 1e-4 |
| Classifier learning rate | CIFAR-10: 5e-4; EMNIST: 1e-4 |
| Autoencoder learning rate / latent dimension | 5e-4 / 128 |
| Validation holdout | Approximately 10% within each client's training classes |
| Final evaluation | Official test split |

One method/variant has **30 experiments**. The ablation notebook has **120**, including the full control. To run a smaller batch, select one official seed and concentration while keeping both datasets, for example `ALPHAS = [0.3]` and `SEEDS = [42]`. Every selected real-data experiment still uses 100 rounds. Restore the accumulated outputs and repeat the remaining batches to complete the grid.

FedAC uses the same client sample indices, validation holdout, classifier initialization, minibatch/augmentation streams and full-participation schedule as LatentFed for each dataset, concentration and seed. The original FedAC local regularization and server operations are retained. Each manifest records hashes of the partition, initialization and schedule so that the common protocol is inspectable.

Generated results are identified by `matched_latentfed_v1` in their manifests. Earlier independently partitioned baseline archives belong to their original allocation protocol; the released notebooks generate the matched-protocol comparison.

## Data

CIFAR-10 downloads the original Python archive from the pinned Hugging Face mirror `MIT-OL-AI-D/cifar-10-python` at revision `a48007227f9e2cd0af96175f4afb5ac4965e261b`. The archive MD5 is `c58f30108f718f92721af3b95e74349a`; torchvision also verifies individual members. Official sample ordering is retained.

EMNIST uses torchvision's **Balanced** split with 47 classes. Both train and test images receive the orientation correction and normalization. Validation samples are excluded from local optimization; summaries and assignments use training data. Internet is needed for initial downloads; cached data is reused. No Kaggle Add Input dataset is required.

## Ablation definitions

| Variant | Autoencoder | MDFE | CALD |
| --- | --- | --- | --- |
| `full` | Convolutional | Enabled | Fixed 64/64 split |
| `no_mdfe` | Convolutional | Identity bypass | Enabled |
| `no_cald` | Convolutional | Enabled | Disabled |
| `ae_only` | **Fully connected** | Absent | Disabled |

The AE-only encoder is `Flatten → Linear(input,512) → ReLU → Linear(512,128)`. Its decoder is `Linear(128,512) → ReLU → Linear(512,input) → Reshape`. The input has 3072 coordinates for CIFAR-10 and 784 for EMNIST; reconstruction output is linear.

All four variants retain classifier cross-entropy and the base AE objective: reconstruction MSE with weight 1.0 plus the class-structured term with weight 0.05. Disabling CALD zeros the invariance, clustering, margin and routing terms. The federated clustering and classifier aggregation pipeline remains active. Full CALD uses a 20-round warm-up and 20-round ramp; AE pretraining rounds are zero.

## Outputs and resuming

Each notebook writes to its own directory under `/kaggle/working` and separates real-data `paper` runs from synthetic `smoke` runs. A run contains configuration, manifest, round/client CSVs, structured diagnostics and a local checkpoint. `RESUME = True` resumes matching checkpoints or reuses completed runs. Source definitions, scientific settings and active-run data/runtime identity are checked.

The final ZIP includes metrics, manifests, configuration, seed summaries and coverage. Set `INCLUDE_CHECKPOINTS_IN_ZIP = True` to include checkpoints before downloading for a later session. Restore the same output hierarchy and compatible runtime to resume. Raw datasets are cached separately and excluded from the ZIP. The full and ablation notebooks use separate output roots; the full control in the ablation notebook is paired with its variants.

W-Acc pools correct/total test counts; M-Acc averages the 20 client accuracies. MC/BC/WC are the mean/best/worst pooled cluster accuracies. Seed endpoints average **rounds 91–100**. The five-seed CSV reports the mean and **sample standard deviation (ddof=1)** across the five endpoints, with accuracy values in percent. Round CSV accuracies are fractions. Missing metrics remain missing; incomplete seed groups do not produce five-seed publication rows.

## Short execution check

Set `SMOKE_TEST = True` in the configuration cell to exercise the complete pipeline on synthetic images for both dataset shapes. This uses two rounds, seed 7, 20/20 clients and a shortened CALD warm-up. Its outputs are labelled synthetic and excluded from publication aggregation. Real-data defaults remain enabled in the distributed notebooks.

All three notebooks were executed top to bottom in isolated folders with project imports blocked: two dataset-shape runs for the full method, eight for the ablations and two for FedAC. Seven automated checks covered standalone execution, the grid/budget, shared partitions/initialization/schedules, FC architecture and ablation losses, checkpoint resume, five-seed aggregation, and agreement of short trajectories with the supplied implementation. Full 100-round real-data GPU grids were not rerun in this validation.

## Uploading to GitHub

Upload these **three notebooks**, this README and the existing Apache-2.0 [LICENSE](LICENSE) to the repository root. Each `.ipynb` file also works individually on Kaggle. No `src/` folder, Python package or build script is needed for execution.

Scientific definitions were taken from the supplied revision notebooks; their original file fingerprints are recorded in notebook metadata. The code has been organized into readable cells, the fully connected AE-only architecture is explicit, and the common data/budget protocol is shared with FedAC. The existing [LatentFed repository](https://github.com/vialt8/LatentFed) license is retained.
