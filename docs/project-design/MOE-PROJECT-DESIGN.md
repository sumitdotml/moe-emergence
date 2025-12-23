# Mixture of Experts Portfolio Project: Design Review & Final Implementation Plan

## Executive Summary

This document captures the complete design review process for a Mixture of Experts (MoE) portfolio project, conducted through a multi-model discussion between Claude Opus 4.5, GPT-5.2, and the project author. The discussion began with a review of an initial plan to integrate MoE layers into GPT-2 and evolved through several rounds of critical feedback, corrections, and refinements.

The final outcome is a scientifically defensible experimental design that addresses the major methodological concerns a rigorous reviewer (at the level of Karpathy or Vaswani) might raise, while remaining feasible within an approximately $80 GPU budget.

---

## Part 1: The Original Plan

### Initial Approach

The original plan proposed the following approach:

The project would use GPT-2 small (124M parameters) as the base model. The dense feed-forward networks (FFNs) in all 12 transformer layers would be replaced with custom MoE layers. Each MoE layer would contain 8 experts with top-2 routing. The experts would use SwiGLU activation (different from GPT-2's original GELU). The router and expert weights would be randomly initialized. Training would occur on a mixed dataset of code, math, and prose (approximately 30MB total). The goal was to demonstrate emergent expert specialization across domains.

### Stated Rationale

The original rationale for using GPT-2 with continued pre-training rather than training from scratch centered on several points. Training a language model from scratch requires enormous compute, making it infeasible on a limited budget. By starting with GPT-2's pretrained attention and embeddings, the model already understands language fundamentals. Expert specialization patterns would emerge faster because the model doesn't need to learn basic language structure. The approach was framed as "pedagogically honest" since the project acknowledges using transfer learning.

---

## Part 2: Initial Critique (Claude Opus 4.5)

### The Core Concern

While the GPT-2 choice was described as "defensible," it was identified as not the "strongest" choice for impressing technical experts who have deep experience with these architectures. Several potential concerns were raised that a rigorous scientist might identify.

### Specific Issues Identified

**Architectural Datedness**: GPT-2 is from 2019, and the field has evolved significantly. A technical expert's first reaction upon seeing "GPT-2" would likely be to question why this particular architecture was chosen.

**Confounding Variables**: The original plan changed two things simultaneously when replacing the FFN: the routing mechanism AND the activation function (GELU to SwiGLU). This makes it impossible to attribute observed effects cleanly to MoE routing versus the activation change.

**Hidden Dimension Mismatch**: GPT-2's FFN uses a 4x expansion factor (768 → 3072 → 768), but SwiGLU typically uses approximately 2.67x to maintain similar parameter counts due to its three-matrix structure. This architectural mismatch wasn't addressed.

**Missing Compute Analysis**: The entire value proposition of MoE is conditional computation, yet the original plan included no measurement of FLOPs, throughput, or memory usage. Without this, the project misses the core argument for why MoE exists.

**No Ablation Studies**: The plan had essentially one experiment (train MoE-GPT-2, look at specialization). A rigorous project would compare MoE vs dense at matched FLOPs, different expert counts, different top-k values, and with/without load balancing.

**Surface-Level Analysis**: Domain-level specialization (code vs math vs prose) is interesting but somewhat expected. More compelling would be analyzing token-type routing within domains.

### Factual Error in Initial Critique

The initial critique incorrectly claimed that GPT-2 uses post-normalization (LayerNorm after attention/MLP). This was factually wrong. GPT-2 actually uses pre-LayerNorm, where normalization happens before the attention and MLP sublayers. The block structure is `x + Attention(LayerNorm(x))` and `x + MLP(LayerNorm(x))`. This error was caught and corrected in the subsequent discussion.

### Initial Recommendations

Three options were proposed with increasing effort and impact levels.

**Option A (Minimal Change)**: Keep GPT-2 but add defensive framing, compute efficiency measurements, and at least one ablation.

**Option B (Moderate Change)**: Switch to Pythia (EleutherAI's more modern architecture) as the base model.

**Option C (Maximum Impact)**: Build a tiny Llama-style architecture from scratch with RMSNorm, RoPE, and SwiGLU.

---

## Part 3: GPT-5.2's First Response

### Corrections to Initial Critique

GPT-5.2 provided several important corrections and refinements.

**Pre-Norm Correction**: Confirmed that the claim about GPT-2 using post-norm was incorrect. GPT-2 blocks are pre-LN, so "outdated because post-norm" is not a fair critique.

**Budget Reality Check**: Building a tiny Llama from scratch is high risk under $80. The budget would be consumed fighting training stability, tokenization, and data pipeline issues, potentially without getting interpretable specialization beyond noise.

**Framing Rebuttal**: "GPT-2 is old" isn't a dealbreaker if properly framed as a controlled intervention baseline with strong ablations and measurements. Scientists care more about experimental cleanliness than backbone recency.

### Critical New Insights

**Warm-Start Requirement**: The original plan to delete pretrained MLPs and replace with random experts is problematic. The correct approach is to warm-start by making experts copies of the pretrained MLP (or at least Expert-0 = original MLP) and bias routing initially.

**Minimal Ablation Set**: With limited budget, the essential ablations are load-balancing on/off, top-1 vs top-2, and potentially 4 vs 8 experts in only the last N layers.

**Modernity Signaling Solution**: Do GPT-2 as the main controlled study, then one confirmatory run on a more modern checkpoint (Pythia-160M-class) to show the phenomenon isn't GPT-2-specific.

---

## Part 4: Refined Design Discussion

### The Warm-Start Deep Dive

The warm-start recommendation was identified as crucial. Here's why it matters:

The original plan initialized MoE layers with random router and expert weights while keeping pretrained attention. This creates a mismatch because the attention layers have learned to produce representations that expect a specific transformation from the FFN. Random experts can't provide this, forcing the model to simultaneously learn routing, expert computations, and re-adapt attention layers.

The warm-start approach initializes all experts as copies of the original pretrained MLP. At initialization, every expert can do exactly what the original FFN did. The model starts in a working state, and as training proceeds, experts gradually specialize from a common starting point. This is much easier to interpret because you observe divergence from a known baseline rather than convergence from random chaos.

### MoE Only in Last N Layers

This recommendation was validated on both practical and scientific grounds.

**Practical**: Reduces training compute by approximately 2/3 compared to MoE in all layers, leaving more budget for ablations.

**Scientific**: Early transformer layers learn generic features (basic syntax, common patterns) that may not benefit from expert specialization. Later layers learn more specific features (semantic relationships, domain-specific patterns) where specialization is most meaningful.

**Recommendation**: MoE in last 4 layers (out of 12) as the default configuration.

### Honest Compute Framing

A critical framing issue was identified: At small scale, MoE may actually be slower than dense due to routing overhead (computing scores, top-k selection, gathering/scattering tokens). This is not a failure but a known property of MoE.

The correct framing is "same active compute, higher capacity via conditional computation" rather than "more efficient than GPT-2." The project should honestly report if MoE is slower at this scale and explain why efficiency gains are more pronounced at larger scales.

---

## Part 5: Symmetry Breaking Discussion

### The Problem

When experts are initialized as exact copies of the pretrained MLP, what drives them to diverge? If all experts are identical and the router has no reason to prefer one over another, specialization may emerge slowly or arbitrarily.

### Mechanisms Discussed

**Per-Expert Weight Noise**: Add small random perturbations to each expert's weights after copying. This gives each expert a slightly different starting point in weight space.

**Router Noise (NoisyTop-k)**: Add noise to routing logits during training. Even if experts are identical, different experts get selected for different tokens due to noise, creating different gradient updates.

**Per-Expert Bias**: Initialize the router with small random biases toward different experts, creating an initial preference structure.

### Noise Scale Calibration

An important correction was made regarding noise magnitude.

**Initial Suggestion**: 0.01 × weight norm (too aggressive)

**Corrected Recommendation**: 1e-3 to 1e-4 × weight std

The distinction matters significantly. For a matrix with dimensions like GPT-2's MLP projection (768 × 3072), the Frobenius norm is much larger than element-wise std due to summing over approximately 2.4 million entries. Using norm instead of std could give perturbations 100× larger than intended, corrupting pretrained representations.

### Final Symmetry Breaking Recommendation

Use minimal mechanisms: tiny expert perturbations (1e-3 × std) combined with NoisyTop-k with annealing. Don't add extra random router biases unless experts fail to diverge.

### Noise Annealing

Router noise should be annealed to zero over the first 20-30% of training. This ensures experts diverge during the critical early phase, then allows the router's learned preferences to dominate. If noise continues throughout training and you report "entropy decreased," a reviewer might question how much is annealing vs actual learned confidence.

### Clean Entropy Logging

When using noisy routing during training, compute entropy on clean logits (no noise) for logging purposes. This separates "training exploration" from "reported router confidence."

---

## Part 6: Additional Stabilization (Z-Loss)

### The Problem Load Balancing Doesn't Solve

Load-balancing loss encourages even utilization of experts but doesn't constrain router logit magnitude. During training, logits can drift to large values, causing several problems: extremely peaked softmax kills exploration, gradients become numerically unstable, and experts with very negative logits become effectively dead even if load balancing tries to revive them.

### Z-Loss Solution

Add a penalty on the log-sum-exp of router logits:

```
z_loss = mean(logsumexp(router_logits, dim=-1) ** 2)
```

This penalizes large logsumexp values, which occur when router logits are extreme. The coefficient is typically 1e-3 to 1e-2. This is a common stabilizer in modern MoE training (used in ST-MoE) and signals familiarity with router failure modes.

### Alternative: Logit Clipping

Clamp router logits to [-C, C] (often C=10-20) before computing softmax and top-k. Less principled but often works in practice.

**Recommendation**: Implement z-loss for the project since it's more principled and citable.

---

## Part 7: Fine-Grained Token Analysis

### The BPE Challenge

GPT-2's BPE tokenizer doesn't tokenize at word boundaries. It learns subword units, so common words might be single tokens while rare words get split. For Python code, tokenization can be unintuitive: `def` might be token 4299, but `define` might split differently, and the token `def` appearing in `define` isn't the same as the standalone keyword.

This means you can't simply say "token ID 4299 is the Python keyword def" and track its routing. Context matters.

### Recommended Approach

**Option 1 (Full Alignment)**: Record token IDs, decoded strings, and surrounding context for each routing decision. Reconstruct original text and align tokens to source positions. For Python code, use Python's `tokenize` module on original source to identify what each span actually is (keyword, identifier, operator, string literal, comment). Aggregate statistics by source-level category rather than token ID.

**Option 2 (Clean Subset)**: Focus on unambiguous tokens that are reliably interpretable regardless of context: newlines, common operators (`=`, `+`), parentheses, brackets, common keywords that don't appear as substrings. Acknowledge this doesn't capture the full picture but ensures unambiguous analysis.

This analysis happens post-training on CPU, so it doesn't consume GPU budget.

---

## Part 8: Final Consensus Design

After multiple rounds of discussion, both models converged on a design that addresses major methodological concerns while remaining feasible under budget constraints.

### Architecture Decisions

**Base Model**: GPT-2 small (124M parameters)

**MoE Placement**: Last 4 transformer layers only (layers 8-11)

**Expert Configuration**: 8 experts with top-1 routing (top-2 as ablation)

**Expert Architecture**: Identical to GPT-2's original MLP (768 → 3072 → 768, GELU activation). Do NOT change to SwiGLU in main study.

### Initialization Protocol

**Expert Weights**: Copy pretrained GPT-2 MLP weights to all 8 experts, then add Gaussian noise with std = 1e-3 × weight std

**Router**: Random initialization (small values), NoisyTop-k during training with noise annealed to zero over first 20-30% of training steps

### Training Configuration

**Dataset**: Mixed code, math, prose (approximately 10MB each, 30MB total)

**Loss Function**: LM loss + load-balancing auxiliary loss (coefficient ~0.01) + z-loss (coefficient ~1e-3)

**Logging**: Total loss, LM loss, load-balance loss, z-loss, per-layer expert utilization, per-layer router entropy (computed on clean logits)

### Ablations (Priority Order)

1. Load-balancing on vs off (demonstrates collapse prevention)
2. Top-1 vs top-2 routing (tests value of multiple experts per token)
3. Pythia-160M confirmatory run (shows generalization beyond GPT-2)
4. 4 vs 8 experts (if budget allows)

### Post-Training Analysis

**Domain-Level**: Expert activation heatmaps (experts × domains)

**Fine-Grained**: Token-type routing analysis with proper BPE handling

**Dynamics**: Router entropy over training visualization

### Measurements to Report

**Compute**: FLOPs per token (MoE vs dense), analytical calculation

**Memory**: Peak GPU memory during training

**Throughput**: Tokens per second, with honest discussion of small-scale overhead

**Parameters**: Total vs active parameters, clearly distinguished

### Framing for Writeup

Frame GPT-2 as a controlled intervention baseline, not a state-of-the-art choice. The claim is "same active compute, higher capacity via conditional computation" not "MoE is more efficient." Report wall-clock overhead honestly and explain that efficiency gains are more pronounced at scale.

---

## Part 9: Detailed Implementation Plan

### Phase 1: MoE Layer Implementation (Days 1-3)

#### Task 1.1: Expert Module

Create an Expert class that mirrors GPT-2's MLP exactly:

```python
# Pseudocode structure
class Expert(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        # Two linear layers with GELU activation (matching GPT-2)
        self.c_fc = nn.Linear(hidden_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))
```

**Deliverable**: Expert module that can be initialized from GPT-2 MLP weights

**Verification**: Forward pass produces identical output to original GPT-2 MLP when weights are copied

#### Task 1.2: Router with NoisyTop-k

Create a Router class with the following features:

```python
# Key components
class Router(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=1, noise_std=0.1):
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.noise_std = noise_std  # Will be annealed
        self.training_step = 0
        self.anneal_steps = 0  # Set during training setup
    
    def forward(self, x):
        logits = self.gate(x)
        
        # Clean logits for entropy logging
        clean_probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(clean_probs * torch.log(clean_probs + 1e-9), dim=-1)
        
        # Noisy logits for selection (during training, before anneal complete)
        if self.training and self.training_step < self.anneal_steps:
            noise_scale = self.noise_std * (1 - self.training_step / self.anneal_steps)
            noise = torch.randn_like(logits) * noise_scale
            logits = logits + noise
        
        # Top-k selection
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Renormalize
        
        return weights, indices, entropy, clean_probs
```

**Deliverable**: Router module with NoisyTop-k and clean entropy logging

**Verification**: Entropy decreases over simulated "training steps" as noise anneals

#### Task 1.3: Load Balancing Loss

Implement the auxiliary loss for balanced expert utilization:

```python
def load_balancing_loss(router_probs, expert_indices, num_experts):
    # router_probs: [batch, seq, num_experts] - softmax probabilities
    # expert_indices: [batch, seq, top_k] - selected expert indices
    
    # Fraction of tokens routed to each expert
    # (using one-hot of selected experts)
    expert_mask = F.one_hot(expert_indices, num_experts).float()
    tokens_per_expert = expert_mask.sum(dim=[0, 1, 2])  # [num_experts]
    fraction_tokens = tokens_per_expert / tokens_per_expert.sum()
    
    # Mean probability assigned to each expert
    mean_prob = router_probs.mean(dim=[0, 1])  # [num_experts]
    
    # Load balancing loss (from Switch Transformer)
    # Encourages fraction_tokens ≈ mean_prob ≈ 1/num_experts
    lb_loss = num_experts * (fraction_tokens * mean_prob).sum()
    
    return lb_loss
```

**Deliverable**: Load balancing loss function

**Verification**: Loss is minimized when expert usage is uniform

#### Task 1.4: Z-Loss

Implement router logit stabilization:

```python
def z_loss(router_logits):
    # router_logits: [batch, seq, num_experts]
    logsumexp = torch.logsumexp(router_logits, dim=-1)  # [batch, seq]
    return torch.mean(logsumexp ** 2)
```

**Deliverable**: Z-loss function

**Verification**: Loss increases with larger logit magnitudes

#### Task 1.5: Complete MoE Layer

Combine all components:

```python
class MoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=8, top_k=1):
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Router(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # [batch*seq, hidden]
        
        weights, indices, entropy, probs = self.router(x_flat)
        
        # Compute expert outputs (efficient batched approach)
        # ... implementation details ...
        
        return output, {
            'entropy': entropy,
            'router_probs': probs,
            'expert_indices': indices,
            'router_logits': self.router.gate(x_flat)
        }
```

**Deliverable**: Complete MoE layer with auxiliary outputs for loss computation and logging

**Verification**: Output shape matches input shape; auxiliary outputs are correctly structured

---

### Phase 2: GPT-2 Integration (Days 4-5)

#### Task 2.1: Load and Inspect GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Inspect structure
for name, module in model.named_modules():
    if 'mlp' in name.lower():
        print(name, type(module))
```

**Goal**: Understand exactly where MLP modules are located and their interface

#### Task 2.2: Create MoE-GPT-2 Wrapper

```python
class MoEGPT2(nn.Module):
    def __init__(self, base_model, moe_layers=[8, 9, 10, 11], num_experts=8, top_k=1):
        super().__init__()
        self.model = base_model
        self.moe_layers = moe_layers
        self.moe_modules = nn.ModuleDict()
        
        for layer_idx in moe_layers:
            # Get original MLP
            original_mlp = self.model.transformer.h[layer_idx].mlp
            
            # Create MoE layer
            moe = MoELayer(
                hidden_size=768,
                intermediate_size=3072,
                num_experts=num_experts,
                top_k=top_k
            )
            
            # Warm-start: copy weights to all experts with small perturbations
            for expert in moe.experts:
                expert.c_fc.weight.data = original_mlp.c_fc.weight.data.clone()
                expert.c_fc.bias.data = original_mlp.c_fc.bias.data.clone()
                expert.c_proj.weight.data = original_mlp.c_proj.weight.data.clone()
                expert.c_proj.bias.data = original_mlp.c_proj.bias.data.clone()
                
                # Add small perturbation (1e-3 × weight std)
                for param in expert.parameters():
                    noise = torch.randn_like(param) * param.std() * 1e-3
                    param.data += noise
            
            self.moe_modules[str(layer_idx)] = moe
        
        # Remove original MLPs from computation (but keep for reference)
        # ... implementation to redirect forward pass ...
```

**Deliverable**: MoE-GPT-2 model with warm-started experts in last 4 layers

**Verification**: Forward pass works; output shapes are correct; sanity check that generation still produces somewhat coherent text (it should, since warm-start preserves functionality)

#### Task 2.3: Forward Pass Modification

Modify the forward pass to use MoE layers and collect auxiliary outputs:

```python
def forward(self, input_ids, attention_mask=None):
    # Standard transformer forward with MoE substitution
    hidden_states = self.model.transformer.wte(input_ids)
    hidden_states = hidden_states + self.model.transformer.wpe(position_ids)
    
    all_aux = []
    
    for idx, block in enumerate(self.model.transformer.h):
        # Attention (unchanged)
        attn_output = block.attn(block.ln_1(hidden_states))
        hidden_states = hidden_states + attn_output
        
        # MLP or MoE
        if idx in self.moe_layers:
            moe = self.moe_modules[str(idx)]
            mlp_output, aux = moe(block.ln_2(hidden_states))
            aux['layer'] = idx
            all_aux.append(aux)
        else:
            mlp_output = block.mlp(block.ln_2(hidden_states))
        
        hidden_states = hidden_states + mlp_output
    
    # Final layer norm and LM head
    hidden_states = self.model.transformer.ln_f(hidden_states)
    logits = self.model.lm_head(hidden_states)
    
    return logits, all_aux
```

**Deliverable**: Modified forward pass that returns logits and MoE auxiliary information

---

### Phase 3: Dataset Preparation (Days 6-7)

#### Task 3.1: Data Collection

**Code (approximately 10MB)**:
- Source: GitHub Python repositories (use GitHub API or existing datasets like CodeParrot)
- Filter: Clean, well-documented functions
- Processing: Extract individual functions/classes, remove very long files

**Math (approximately 10MB)**:
- Source: MATH dataset, GSM8K, or educational math content
- Content: Problems with solutions, proofs, explanations
- Processing: Ensure natural language explanations, not just equations

**Prose (approximately 10MB)**:
- Source: OpenWebText subset, WikiText, or similar
- Content: Articles, essays, stories
- Processing: Filter for quality, remove boilerplate

#### Task 3.2: Tokenization and DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MixedDomainDataset(Dataset):
    def __init__(self, code_texts, math_texts, prose_texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Combine with domain labels (for analysis, not training)
        self.samples = []
        for text in code_texts:
            self.samples.append({'text': text, 'domain': 'code'})
        for text in math_texts:
            self.samples.append({'text': text, 'domain': 'math'})
        for text in prose_texts:
            self.samples.append({'text': text, 'domain': 'prose'})
        
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'domain': sample['domain']  # For analysis only
        }
```

**Deliverable**: Balanced dataset with domain labels for post-hoc analysis

**Verification**: Approximately equal token counts across domains; sample inspection for quality

#### Task 3.3: Held-Out Test Sets

Create separate held-out sets for each domain (not seen during training) for post-training analysis. Aim for approximately 100 samples per domain.

---

### Phase 4: Training Infrastructure (Days 8-9)

#### Task 4.1: Training Loop

```python
def train_moe_gpt2(model, train_loader, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    
    # Set anneal steps for router noise
    for layer_idx in model.moe_layers:
        model.moe_modules[str(layer_idx)].router.anneal_steps = int(config.total_steps * 0.25)
    
    step = 0
    for epoch in range(config.epochs):
        for batch in train_loader:
            # Forward pass
            logits, aux_outputs = model(batch['input_ids'].to(device))
            
            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['input_ids'][..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Auxiliary losses
            total_lb_loss = 0
            total_z_loss = 0
            for aux in aux_outputs:
                total_lb_loss += load_balancing_loss(
                    aux['router_probs'],
                    aux['expert_indices'],
                    model.num_experts
                )
                total_z_loss += z_loss(aux['router_logits'])
            
            total_lb_loss /= len(aux_outputs)
            total_z_loss /= len(aux_outputs)
            
            # Combined loss
            loss = lm_loss + config.lb_coef * total_lb_loss + config.z_coef * total_z_loss
            
            # Backward and step
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging (see Task 4.2)
            ...
            
            # Update router training step (for noise annealing)
            for layer_idx in model.moe_layers:
                model.moe_modules[str(layer_idx)].router.training_step = step
            
            step += 1
```

**Deliverable**: Complete training loop with all loss components

#### Task 4.2: Logging Infrastructure

```python
# Metrics to log every N steps
metrics = {
    'total_loss': loss.item(),
    'lm_loss': lm_loss.item(),
    'lb_loss': total_lb_loss.item(),
    'z_loss': total_z_loss.item(),
}

# Per-layer metrics
for aux in aux_outputs:
    layer = aux['layer']
    
    # Expert utilization (how many tokens went to each expert)
    expert_counts = torch.bincount(
        aux['expert_indices'].view(-1),
        minlength=model.num_experts
    ).float()
    expert_fractions = expert_counts / expert_counts.sum()
    
    metrics[f'layer_{layer}_expert_utilization'] = expert_fractions.cpu().numpy()
    metrics[f'layer_{layer}_utilization_std'] = expert_fractions.std().item()
    
    # Router entropy (computed on clean logits)
    metrics[f'layer_{layer}_mean_entropy'] = aux['entropy'].mean().item()

# Log to wandb/tensorboard/csv
logger.log(metrics, step=step)
```

**Deliverable**: Comprehensive logging for all relevant metrics

#### Task 4.3: Checkpointing

Save checkpoints every 1000 steps and at end of training. Include model weights, optimizer state, and training step (for resumption and for noise annealing state).

#### Task 4.4: Compute Measurements

```python
# FLOPs calculation (analytical)
def compute_flops_per_token(model, moe_layers, num_experts, top_k):
    # Attention FLOPs (unchanged)
    attn_flops = ...  # Standard calculation
    
    # MLP FLOPs
    dense_mlp_flops = 2 * 768 * 3072 + 2 * 3072 * 768  # per layer per token
    
    # MoE FLOPs = router + top_k * expert_mlp
    router_flops = 2 * 768 * num_experts
    moe_flops_per_layer = router_flops + top_k * dense_mlp_flops
    
    # Total
    moe_layer_count = len(moe_layers)
    dense_layer_count = 12 - moe_layer_count
    
    total_mlp_flops = (dense_layer_count * dense_mlp_flops + 
                       moe_layer_count * moe_flops_per_layer)
    
    return total_attn_flops + total_mlp_flops

# Memory and throughput (empirical)
# Measure during training with torch.cuda.max_memory_allocated()
# Measure tokens/second as batch_size * seq_len / step_time
```

**Deliverable**: FLOPs calculation and empirical measurements during training

---

### Phase 5: Training Execution (Days 10-12)

#### Task 5.1: Main Training Run

Configuration for main experiment:
- Batch size: 8-16 (depending on GPU memory)
- Sequence length: 512
- Learning rate: 5e-5 to 1e-4
- Warmup: 10% of total steps
- Load balancing coefficient: 0.01
- Z-loss coefficient: 0.001
- Total steps: 10,000-20,000 (adjust based on budget)

**Estimated cost**: $30-50 depending on GPU choice and step count

**Monitor during training**: Loss curves should decrease; expert utilization should be roughly balanced (no expert getting <5% or >30% of tokens); router entropy should decrease over time

#### Task 5.2: Load Balancing Ablation

Run a shortened version (maybe 5,000 steps) with load balancing coefficient = 0.

**Expected result**: Expert collapse (1-3 experts handling >80% of tokens)

**Purpose**: Empirically demonstrates why load balancing is necessary

**Estimated cost**: $10-15

#### Task 5.3: Top-k Ablation (if budget allows)

Run with top_k=2 instead of top_k=1 for similar step count.

**Purpose**: Shows effect of routing to multiple experts per token

**Estimated cost**: $15-20

#### Task 5.4: Pythia Confirmatory Run (if budget allows)

Repeat main experiment on Pythia-160M to show generalization.

**Estimated cost**: $15-20

---

### Phase 6: Post-Training Analysis (Days 13-15)

#### Task 6.1: Domain-Level Expert Activation

```python
def analyze_domain_routing(model, test_loaders, device):
    """
    test_loaders: {'code': loader, 'math': loader, 'prose': loader}
    """
    model.eval()
    
    # Collect routing decisions per domain
    domain_routing = {domain: [] for domain in test_loaders.keys()}
    
    with torch.no_grad():
        for domain, loader in test_loaders.items():
            for batch in loader:
                _, aux_outputs = model(batch['input_ids'].to(device))
                
                for aux in aux_outputs:
                    # Record which experts were selected
                    domain_routing[domain].append({
                        'layer': aux['layer'],
                        'expert_indices': aux['expert_indices'].cpu(),
                        'router_probs': aux['router_probs'].cpu()
                    })
    
    # Aggregate into expert activation frequencies per domain per layer
    activation_matrix = {}  # [layer][domain][expert] -> frequency
    
    # ... aggregation logic ...
    
    return activation_matrix
```

**Deliverable**: Expert activation frequencies broken down by domain and layer

#### Task 6.2: Fine-Grained Token Analysis

```python
import tokenize
import io

def analyze_token_types_python(model, code_samples, tokenizer, device):
    """
    Analyze routing patterns for different Python token types
    """
    model.eval()
    results = []
    
    for code in code_samples:
        # Get model routing decisions
        encoding = tokenizer(code, return_tensors='pt')
        _, aux_outputs = model(encoding['input_ids'].to(device))
        
        # Decode each BPE token and its position in original text
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Use Python tokenizer on original code
        try:
            python_tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except:
            continue
        
        # Align BPE tokens to Python token types
        # This is the tricky part - need to map character positions
        # ... alignment logic ...
        
        # Record routing by Python token type
        for bpe_idx, python_type in alignments:
            expert_idx = aux_outputs[layer_idx]['expert_indices'][0, bpe_idx].item()
            results.append({
                'python_type': python_type,  # e.g., NAME, KEYWORD, OP, STRING
                'expert': expert_idx,
                'layer': layer_idx
            })
    
    return results
```

**Deliverable**: Routing statistics broken down by token type within domains

#### Task 6.3: Router Entropy Dynamics

Extract entropy values from training logs and create visualization showing how router confidence evolves during training, potentially broken down by layer.

---

### Phase 7: Visualization (Days 16-17)

#### Task 7.1: Expert-Domain Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_expert_domain_heatmap(activation_matrix, layer_idx):
    """
    Create heatmap showing expert activation frequency by domain
    """
    # activation_matrix[layer][domain][expert] -> frequency
    
    data = []
    for domain in ['code', 'math', 'prose']:
        row = [activation_matrix[layer_idx][domain][e] for e in range(8)]
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        data,
        xticklabels=[f'E{i}' for i in range(8)],
        yticklabels=['Code', 'Math', 'Prose'],
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        ax=ax
    )
    ax.set_xlabel('Expert')
    ax.set_ylabel('Domain')
    ax.set_title(f'Expert Activation Frequency - Layer {layer_idx}')
    
    return fig
```

**Deliverable**: Publication-quality heatmaps for each MoE layer

#### Task 7.2: Router Entropy Over Training

```python
def plot_entropy_dynamics(training_logs):
    """
    Plot how router entropy changes during training
    """
    steps = [log['step'] for log in training_logs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for layer in [8, 9, 10, 11]:
        entropies = [log[f'layer_{layer}_mean_entropy'] for log in training_logs]
        ax.plot(steps, entropies, label=f'Layer {layer}')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Router Entropy')
    ax.set_title('Router Confidence Over Training')
    ax.legend()
    ax.axvline(x=steps[-1]*0.25, color='gray', linestyle='--', 
               label='Noise annealing complete')
    
    return fig
```

**Deliverable**: Entropy dynamics plot showing increasing router confidence

#### Task 7.3: Token-Type Routing Visualization

```python
def plot_token_type_routing(token_analysis_results, layer_idx):
    """
    Show how different Python token types route to experts
    """
    # Aggregate by token type
    type_expert_counts = defaultdict(lambda: defaultdict(int))
    for result in token_analysis_results:
        if result['layer'] == layer_idx:
            type_expert_counts[result['python_type']][result['expert']] += 1
    
    # Normalize to frequencies
    # ... normalization logic ...
    
    # Create grouped bar chart or heatmap
    # ... plotting logic ...
```

**Deliverable**: Visualization of token-type routing patterns

#### Task 7.4: Collapse Comparison Visualization

Side-by-side comparison of expert utilization with and without load balancing:

```python
def plot_collapse_comparison(with_lb_stats, without_lb_stats):
    """
    Show expert utilization with vs without load balancing
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Without load balancing
    axes[0].bar(range(8), without_lb_stats)
    axes[0].set_title('Without Load Balancing\n(Expert Collapse)')
    axes[0].set_xlabel('Expert')
    axes[0].set_ylabel('Fraction of Tokens')
    
    # With load balancing
    axes[1].bar(range(8), with_lb_stats)
    axes[1].set_title('With Load Balancing\n(Balanced Utilization)')
    axes[1].set_xlabel('Expert')
    axes[1].set_ylabel('Fraction of Tokens')
    
    return fig
```

**Deliverable**: Clear demonstration of load balancing importance

---

### Phase 8: Documentation (Days 18-20)

#### Task 8.1: GitHub README Structure

```markdown
# Emergent Expert Specialization in Mixture of Experts

## Overview
[Brief description of project and key findings]

## Key Results
[2-3 most compelling visualizations with captions]

## Architecture
[Diagram of MoE-GPT-2 architecture]
[Table: Total params, Active params, FLOPs comparison]

## Methodology
[Brief description of experimental design]
[Justification for key decisions: GPT-2 as controlled baseline, warm-start, last 4 layers, etc.]

## Results

### Expert Specialization by Domain
[Heatmaps and discussion]

### Fine-Grained Token Analysis
[Token-type routing results]

### Training Dynamics
[Loss curves, entropy dynamics]

### Ablation Studies
[Load balancing on/off comparison]
[Top-1 vs Top-2 if completed]

## Compute Efficiency Analysis
[FLOPs comparison table]
[Honest discussion of small-scale overhead]

## Limitations and Future Work
[What this study doesn't show]
[What would be interesting to explore with more compute]

## Reproducing Results
[Environment setup]
[Training commands]
[Expected runtime and cost]

## References
[Key papers: Switch Transformer, ST-MoE, Mixtral, etc.]
```

#### Task 8.2: Code Organization

```
moe-portfolio/
├── README.md
├── moe/
│   ├── __init__.py
│   ├── expert.py         # Expert module
│   ├── router.py         # Router with NoisyTop-k
│   ├── moe_layer.py      # Complete MoE layer
│   └── losses.py         # Load balancing, z-loss
├── model/
│   ├── __init__.py
│   └── moe_gpt2.py       # MoE-GPT-2 integration
├── data/
│   ├── __init__.py
│   └── dataset.py        # Mixed domain dataset
├── training/
│   ├── __init__.py
│   ├── train.py          # Training loop
│   └── config.py         # Hyperparameters
├── analysis/
│   ├── __init__.py
│   ├── domain_routing.py # Domain-level analysis
│   ├── token_analysis.py # Fine-grained analysis
│   └── visualize.py      # All plotting functions
├── scripts/
│   ├── train_main.py     # Main training script
│   ├── train_ablation.py # Ablation runs
│   └── analyze.py        # Post-training analysis
├── notebooks/
│   └── results.ipynb     # Results exploration
└── configs/
    ├── main.yaml
    └── ablation.yaml
```

#### Task 8.3: Blog Post / Project Page

Write a narrative version of the README that tells the story:
- The question: Can experts self-organize without supervision?
- The experimental design: Why these specific choices?
- The findings: What emerged?
- The implications: Why does this matter?

Target length: 1500-2000 words, readable in 10 minutes

---

### Budget Summary

| Component | Estimated Cost | Priority |
|-----------|---------------|----------|
| Main training run (15K steps) | $35-45 | Critical |
| Load balancing ablation (5K steps) | $10-15 | Critical |
| Top-1 vs Top-2 ablation (10K steps) | $15-20 | Important |
| Pythia confirmatory run (10K steps) | $15-20 | Important |
| **Total** | **$75-100** | |

If budget is tight, prioritize: Main run + Load balancing ablation (minimum viable for strong project)

---

### Timeline Summary

| Phase | Days | Key Deliverables |
|-------|------|------------------|
| MoE Implementation | 1-3 | Expert, Router, Losses, MoE Layer |
| GPT-2 Integration | 4-5 | Warm-started MoE-GPT-2 model |
| Dataset Preparation | 6-7 | Balanced training data, held-out test sets |
| Training Infrastructure | 8-9 | Training loop, logging, checkpointing |
| Training Execution | 10-12 | Trained models, ablation results |
| Post-Training Analysis | 13-15 | Routing statistics, token analysis |
| Visualization | 16-17 | Publication-quality figures |
| Documentation | 18-20 | README, code organization, blog post |

**Total: approximately 3 weeks of focused work**

---

## Appendix A: Key Papers to Reference

1. **Switch Transformers** (Fedus et al., 2021) - Load balancing loss formulation
2. **ST-MoE** (Zoph et al., 2022) - Z-loss and router design
3. **Mixtral** (Mistral AI, 2023) - Production MoE architecture
4. **GShard** (Lepikhin et al., 2020) - Early large-scale MoE
5. **Outrageously Large Neural Networks** (Shazeer et al., 2017) - Original MoE for NLP

---

## Appendix B: Potential Reviewer Questions and Responses

**Q: Why GPT-2 instead of a more modern architecture?**

A: GPT-2 serves as a well-understood controlled baseline. Its behavior is thoroughly documented, making it easier to attribute observed effects to our MoE intervention rather than architectural idiosyncrasies. We validate generalization with a Pythia-160M confirmatory run.

**Q: Isn't MoE supposed to be more efficient? Your results show it's slower.**

A: MoE's efficiency advantage is most pronounced at scale, where the routing overhead is amortized over larger expert computations. At our experimental scale, the routing cost dominates. We include this observation as evidence that we understand the true tradeoffs of MoE, not just the marketing pitch.

**Q: How do you know the specialization isn't just random?**

A: We verify through multiple methods: specialization patterns are consistent across held-out samples, different layers show coherent (not random) patterns, and the ablation without load balancing shows qualitatively different (collapsed) behavior, indicating the balanced specialization is a learned phenomenon.

**Q: Why only the last 4 layers?**

A: Research suggests that early transformer layers learn more generic features that may not benefit from expert specialization, while later layers learn more task/domain-specific representations. We chose the last 4 layers as a reasonable default that balances compute cost with opportunity for specialization. Budget permitting, we would explore this as an ablation.

---

## Appendix C: Common Failure Modes and Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| Expert collapse | 1-2 experts handle >80% tokens | Verify load balancing loss is being applied; increase coefficient |
| Router instability | Loss spikes, NaN values | Add z-loss; reduce learning rate |
| No specialization | Experts activate uniformly across domains | Train longer; verify domains are actually different; check warm-start perturbation |
| Memory OOM | CUDA out of memory | Reduce batch size; use gradient checkpointing |
| Slow training | Much slower than expected | Verify efficient expert computation; consider top-1 instead of top-2 |

---

*Document generated from multi-model design review discussion between Claude Opus 4.5 and GPT-5.2*
