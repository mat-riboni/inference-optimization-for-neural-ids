# Efficient Inference for Intrusion Detection Systems

> **⚠️ Maintenance Notice**  
> This repository is currently **under maintenance**.  
> The code was developed during the thesis process and is **not clean, optimized, or well documented**.  
> A future refactor will make it usable... maybe.

---

## Overview

This repository contains the implementation and experiments developed for the Master's thesis:

> **"Optimization of Neural Network Inference Time for Intrusion Detection Systems"**  
> Author: *Mattia Riboni*  
> Supervisor: *Prof. Stefano Iannucci*  
> Co-supervisor: *Simone Albero*  
> Roma Tre University  
> Academic Year 2024/2025

The goal of this work is to reduce the **inference latency** of neural networks used for **Intrusion Detection Systems (IDS)**, while maintaining comparable accuracy to the original full-precision model.

The study focuses on three optimization techniques:
- **Quantization** – reducing numerical precision to accelerate inference.
- **Pruning** – removing redundant neurons and connections.
- **DenseBranchyNet (Early Exit)** – enabling adaptive inference depth based on sample difficulty.

---

## Methodology

### 1. Quantization
Implemented as **post-training dynamic quantization** using PyTorch (`torch.quantization.quantize_dynamic`).  
Two configurations are explored:
- **Full quantization** – all linear layers converted to INT8 except the output layer.  
- **Hidden-only quantization** – the first and last layers kept in FP32 for numerical stability.

Results show a **latency reduction of about 35–40%**, but a strong degradation in classification performance due to heterogeneous feature scales (numeric vs categorical embeddings).

---

### 2. Pruning
Applies **structured post-training pruning** using **L1-norm neuron importance**.  
Entire neurons are removed from the hidden layers, producing a smaller but dense model that runs faster on standard hardware.  
A short fine-tuning phase follows to recover accuracy.

Key findings:
- Keeping **60–70% of neurons** preserves nearly full accuracy.  
- Inference latency improves by up to **2–3×**.  
- Below 40%, the model loses representational capacity and F1 drops sharply.

---

### 3. DenseBranchyNet (Early Exit)
Implements a simplified version of **BranchyNet** adapted to tabular IDS data.  
A single early-exit branch is added after the first hidden layer.  
Inference stops early for samples with high confidence, estimated via:
- **Entropy** of predicted probabilities, or  
- **Margin** between the top two probabilities.

A **temperature scaling** step calibrates probabilities before decision-making.  
With proper threshold selection, this approach reduces average inference time by **up to 50%**, maintaining accuracy close to the baseline.

---

## Dataset

All experiments are based on the **TON_IoT dataset**, created by the *Cyber Range Lab* at the  
**University of New South Wales (UNSW), Canberra**.

- Official page: [https://research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets)  
- Citation:  
  *Ahmad, J. et al. (2021). TON_IoT Datasets: A new generation of datasets for the Internet of Things (IoT) and Industrial IoT (IIoT) networks. UNSW Canberra Cyber.*

This work uses only the **HTTP network traffic subset**, filtered and preprocessed as follows:
- Removal of underrepresented classes (*mitm*, *dos*).  
- Normalization of numerical features and label encoding of categorical ones.  
- Final dataset includes six traffic categories:  
  `normal`, `scanning`, `ddos`, `injection`, `xss`, and `password`.

---

## Results Summary

| Technique | Description | Latency Improvement | Notes |
|------------|-------------|----------------------|-------|
| **Baseline (FP32)** | Full-precision reference model | — | Reference |
| **Quantization (Full INT8)** | All linear layers quantized | 35% | Large degradation on some classes |
| **Quantization (Hidden-only)** | First/last layers in FP32 | 40% | Again large degradation |
| **Structured Pruning (60%)** | 40% neurons removed | 200–300% | Best accuracy/latency balance |
| **DenseBranchyNet (Early Exit)** | Adaptive inference with early stop | 50% | Margin-based exit more efficient |

---

### Contact
**Mattia Riboni**  
📧 [mattiariboni@gmail.com](mailto:mattiariboni@gmail.com)
