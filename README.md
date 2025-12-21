# MoE Emergence

## Objective

To train a small MoE model on 3 distinct domains (code, math, natural language), and create visualizations / consumable artifacts showing:

- Expert specialization emergence - which experts become domain specialists
- The load balancing problem - what happens without auxiliary loss
- Routing patterns - heatmaps of which tokens → which experts

## Progress Log

- [x] FFN (SwiGLU)
- [x] MoE Layer (with Switch/Mixtral-style load balancing)
- [ ] Embedding
- [ ] GQA
- [ ] RoPE
- [ ] Rest of the architecture
- [ ] Training loop
- [ ] Dataset
- [ ] Eval
- [ ] Visualizations

## Formatting

Ruff being used for formatting here.

- **VSCode / Cursor**: Install the Ruff extension. Settings are in `.vscode/settings.json`.
- **Neovim**: Configure Ruff inside nvim itself. For reference, see how I've done [mine](https://github.com/search?q=repo%3Asumitdotml%2Fdotfiles%20ruff&type=code). For formatting, my command is `<leader>gf`.
