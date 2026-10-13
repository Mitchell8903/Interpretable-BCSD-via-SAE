# Interpretable BCSD via Sparse Autoencoders

This project investigates whether the internal representations of a binary code similarity model (BCSD) decompose into sparse, human-readable features. We train sparse autoencoders on the
finetuned per-token activations of [jTrans](https://github.com/vul337/jTrans) and measure whether the resulting features reconstruct jTrans faithfully and are human-interpretable. jTrans itself is never retrained or modified.

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
bash ./scripts/download_data.sh 
```

Download the SAE model checkpoints:

```bash
bash ./scripts/download_checkpoints.sh 
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
