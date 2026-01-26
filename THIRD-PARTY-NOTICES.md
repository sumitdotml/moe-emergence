# Third-Party Notices

This repository depends on the following third-party software, datasets, and academic works.
More will be added as I continue to develop this project.

---

## Software Dependencies

### PyTorch

- **License:** BSD-3-Clause
- **Copyright:** Copyright (c) 2016-present, Facebook Inc.
- **Source:** https://github.com/pytorch/pytorch
- **Usage:** Core deep learning framework

### Hugging Face Transformers

- **License:** Apache License 2.0
- **Copyright:** Copyright 2018-present The Hugging Face team
- **Source:** https://github.com/huggingface/transformers
- **Usage:** GPT-2 model architecture and pretrained weights

### Hugging Face Datasets

- **License:** Apache License 2.0
- **Copyright:** Copyright 2020-present The Hugging Face team
- **Source:** https://github.com/huggingface/datasets
- **Usage:** Data loading and streaming

### NumPy

- **License:** BSD-3-Clause
- **Copyright:** Copyright (c) 2005-2024, NumPy Developers
- **Source:** https://github.com/numpy/numpy
- **Usage:** Numerical operations

---

## Pretrained Models

### GPT-2

- **License:** MIT
- **Copyright:** Copyright (c) 2019 OpenAI
- **Source:** https://huggingface.co/gpt2
- **Paper:** Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
- **Usage:** Base model architecture for MoE integration

---

## Datasets

Datasets are downloaded at runtime from Hugging Face and are not redistributed with this project.

### Verified

#### MathQA

- **License:** Apache License 2.0
- **Copyright:** Allen Institute for AI
- **Source:** https://aclanthology.org/N19-1245/
- **Paper:** Amini et al., "MathQA: Towards Interpretable Math Word Problem Solving" (2019)
- **Usage:** Math domain training data

### Pending Investigation

The following datasets are candidates under evaluation. Final choices will be documented after I do
some sample verification.

#### Code Domain (one of)

- **CodeParrot-clean** (`codeparrot/codeparrot-clean`) - Mixed licenses per file
- **StarCoderData** (`bigcode/starcoderdata`) - Requires TOS acceptance

#### Prose Domain (one of)

- **WikiText-103** (`Salesforce/wikitext`) - CC BY-SA 3.0 / GFDL
- **OpenWebText** (`Skylion007/openwebtext`) - CC0

---

## Academic Works and Algorithms

This project implements techniques from the following academic papers:

### Switch Transformers

- **Paper:** Fedus, Zoph, and Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)
- **Link:** https://arxiv.org/abs/2101.03961
- **Usage:** Load balancing loss formulation

### ST-MoE (Z-Loss)

- **Paper:** Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (2022)
- **Link:** https://arxiv.org/abs/2202.08906
- **Usage:** Router logit stabilization via z-loss

### Noisy Top-k Gating

- **Paper:** Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- **Link:** https://arxiv.org/abs/1701.06538
- **Usage:** Noisy top-k gating mechanism for expert routing

### Mixtral of Experts

- **Paper:** Jiang et al., "Mixtral of Experts" (2024)
- **Link:** https://arxiv.org/abs/2401.04088
- **Usage:** Architecture reference and expert dispatch pattern

### SwiGLU Activation

- **Paper:** Shazeer, "GLU Variants Improve Transformer" (2020)
- **Link:** https://arxiv.org/abs/2002.05202
- **Usage:** SwiGLU FFN reference implementation

### Straight-Through Estimator

- **Paper:** Bengio, Leonard, and Courville, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation" (2013)
- **Link:** https://arxiv.org/abs/1308.3432
- **Usage:** Gradient flow for top-1 discrete routing

---

## Development Tools

### Ruff

- **License:** MIT
- **Source:** https://github.com/astral-sh/ruff
- **Usage:** Code formatting and linting

### ruff-pre-commit

- **License:** MIT OR Apache 2.0 (dual-licensed)
- **Source:** https://github.com/astral-sh/ruff-pre-commit
- **Usage:** Pre-commit hooks for Ruff

---

## License Texts

The full text of referenced licenses can be found at:

- **MIT License:** https://opensource.org/licenses/MIT
- **BSD-3-Clause:** https://opensource.org/licenses/BSD-3-Clause
- **Apache License 2.0:** https://www.apache.org/licenses/LICENSE-2.0
- **CC BY-SA 3.0:** https://creativecommons.org/licenses/by-sa/3.0/
- **CC0:** https://creativecommons.org/publicdomain/zero/1.0/
- **GFDL:** https://www.gnu.org/licenses/fdl-1.3.html
