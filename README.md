# LatentFed
Code for the paper "LatentFed: Reliability-Aware Latent Representations for Clustered Federated Learning under Statistical Heterogeneity"

In this repo you can find the notebook and be able to launch the experiments over 5 seeds "42, 101, 2024, 3407, 8888" and different Dirichlet distributions for the data partitioning following these alphas "0.3, 0.5, 0.7" 

Datasets: "Cifar-10", "EMNIST"

Abstract of the paper:

Federated learning suffers performance degradation under statistical heterogeneity, where clients
follow different data distributions and local models drift toward incompatible optima. Clustered Fed-
erated Learning mitigates this issue by grouping clients with similar distributions, but most existing
approaches infer client similarity from gradients, model parameters, or local losses. These signals are
indirect proxies of the underlying data distribution and can be unstable across communication rounds,
especially under strong non-IID conditions.
This paper proposes LatentFed, a clustered federated learning framework that formulates client
grouping as a latent-representation learning problem. Instead of clustering clients in parameter or
gradient space, each client learns compact dataset-level latent summaries using a convolutional au-
toencoder. The first contribution is a Mean-Distortion Feature Enhancement (MDFE) module, which
introduces channel-wise reliability awareness into the encoder by combining mean activation, peak
response, and distortion information. MDFE suppresses unstable feature channels and improves the
quality of client latent summaries. The second contribution is Cluster-Aware Latent Disentanglement
(CALD), which decomposes latent representations into invariant and cluster-specific components and
uses server-maintained prototypes to stabilize cluster structure over communication rounds.
LatentFed performs density-based clustering on the learned client summaries and trains cluster-
specific federated models using standard cluster-wise aggregation. Experiments on CIFAR-10 and
EMNIST under controlled Dirichlet label-skew settings show that representation-aware latent cluster-
ing improves robustness, convergence stability, and clustering quality compared with conventional FL
and clustered FL baselines. Ablation studies confirm the individual contribution of MDFE and CALD
to producing compact, discriminative, and temporally stable client representations.
