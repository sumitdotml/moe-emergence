# MoE Emergence

## Objective

To train a small MoE model on 3 distinct domains (code, math, natural language), and create visualizations / consumable artifacts showing:

- Expert specialization emergence - which experts become domain specialists
- The load balancing problem - what happens without auxiliary loss
- Routing patterns - heatmaps of which tokens → which experts

## Progress Log

- [x] FFN (SwiGLU)
- [⏳] MoE Layer (mostly done, just need to add load balancing)
- [ ] Embedding
- [ ] GQA
- [ ] RoPE
- [ ] Rest of the architecture
- [ ] Training loop
- [ ] Dataset
- [ ] Eval
- [ ] Visualizations

...