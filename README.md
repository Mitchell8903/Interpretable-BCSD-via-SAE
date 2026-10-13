# Interpretable BCSD via Sparse Autoencoders

This project investigates whether the internal representations of a binary code similarity model (BCSD) decompose into sparse, human-readable features. We train sparse autoencoders on the
finetuned per-token activations of [jTrans](https://github.com/vul337/jTrans) and measure whether the resulting features reconstruct jTrans faithfully and are human-interpretable. jTrans itself is never retrained or modified.

## Sparse Autoencoder

The encoding and decoding compoenents of the forward pass are

```math
\begin{aligned}
z &= \mathrm{ReLU}\Big(\mathrm{TopK}_k\big(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}}\big)\Big) && \in \mathbb{R}^{d_{\text{sae}}} \\
\hat{x} &= W_{\text{dec}} z + b_{\text{pre}} && \in \mathbb{R}^{d_{\text{model}}}
\end{aligned}
```

where $\mathrm{TopK}_n$ keeps a vector's $n$ largest entries and zeroes the rest.

The training loss, over a batch $\{x_1, \dots, x_B\}$ with $z_i$ and
$\hat{x}_i$ from the forward pass above, $D$ the set of dead latents and
$\mathbb{1}_D \in \{0,1\}^{d_{\text{sae}}}$ its indicator:

```math
\begin{aligned}
e_i &= \mathrm{sg}\big(x_i - \hat{x}_i\big) && \in \mathbb{R}^{d_{\text{model}}} \\
z_{\text{aux},i} &= \mathrm{ReLU}\Big(\mathrm{TopK}_{\min(k_{\text{aux}}, |D|)}\big((W_{\text{enc}}(x_i - b_{\text{pre}}) + b_{\text{enc}}) \odot \mathbb{1}_D\big)\Big) && \in \mathbb{R}^{d_{\text{sae}}} \\
\hat{e}_i &= W_{\text{dec}} z_{\text{aux},i} && \in \mathbb{R}^{d_{\text{model}}} \\
\mathcal{L} &= \frac{1}{B} \sum_{i=1}^{B} \Big( \underbrace{\lVert x_i - \hat{x}_i \rVert_2^2}_{\text{reconstruction}} + \alpha \underbrace{\lVert e_i - \hat{e}_i \rVert_2^2}_{\text{AuxK}} \Big), \qquad \alpha = \tfrac{1}{32}, \quad k_{\text{aux}} = 512
\end{aligned}
```

- $x$ is one assembly token's residual state/embedding at layer 6 of jTrans and a function's activation of a feature is its maximum over the function's asm tokens.

- $\mathrm{sg}(\cdot)$ is stop-gradient, so only the dead latents learn from the AuxK term; it is zero when no latent is dead. Evaluation reports the reconstruction term alone.

- Decoder columns are renormalised to unit norm after every step. Encoder weights are initialised as the decoder transpose and $b_{\text{pre}}$ is subtracted before encoding (Bricken et al. 2023). 

- Inputs are rescaled so $\mathbb{E}\lVert x \rVert_2 = \sqrt{d_{\text{model}}}$. 

- Latent features that have not fired for 10M
training tokens are *dead*, and $z_{\text{aux}}$ keeps the top $k_{\text{aux}} = 512$ of their pre-activations to reconstruct the main model's error $e$, with $\alpha = 1/32$.

## Reproducing the experiments

First, clone and setup the environment

```bash
git clone --recurse-submodules https://github.com/Mitchell8903/Interpretable-BCSD-via-SAE.git
cd Interpretable-BCSD-via-SAE
conda env create -f environment.yml
conda activate ibcsd-sae
```

Download the necessary data (model checkpoints, embeddings, BinaryCorp-3M)

```bash
./scripts/download_data.sh 
```

Download the SAE model checkpoints:

```bash
./scripts/download_checkpoints.sh 
```

Reproduce the jTrans baseline recall@1:

```bash
python src/baseline.py
```

Cache the residual activations for each token at a given layer, saved into `data/activation`:

```bash
python src/activations.py --layers 6
```

Train the sparse autoencoder with the specified hyperparameters:

```bash
python src/train_sae.py --layer 6 --d_sae 16384 --k 32
```
*Use `--sweep` to train 3 seeds x `k` {16, 32, 64} x `d_sae` {4096, 8192, 16384} (default x3 seeds at 60 epochs each)*

Evaluate the reconstruction fidelity metrics on a specified SAE configuration:

```bash
python src/eval_recon.py --layer 6 --d_sae 16384 --k 32
```

Evaluate automated interpretability metrics on a specified SAE configuration, and raw/PCA baselines. 
```bash
export OPENAI_API_KEY=...        
python src/eval_interp.py --layer 6       
python src/interp_baselines.py --layer 6
```

*By default, the script uses the OpenAI API and gpt-6-luna, which requries OPENAI_API_KEY to be set. Local Ollama models can be used by specifying an available model (e.g. `--model qwen2.5-coder:14b`).*


Aggregate the results at a layer into a summary (e.g. `data/layer6/summary.json`)
```bash  
python src/aggregate.py --layer 6         # summary.json
```
