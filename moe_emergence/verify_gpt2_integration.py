"""
GPT-2 MoE Integration Verification

This script verifies that the MoE wrapper integrates correctly with
an actual HuggingFace GPT-2 model. This is the last verification before
Phase 3.

Tests:
1. MoE layers installed in correct blocks with expected parameter counts
2. Warm-start parity against baseline GPT-2 MLP outputs
3. Forward pass output shapes
4. Auxiliary routing outputs are consistent (probs, logits, entropy, indices)
5. Load balance and z-loss are finite and within expected ranges
6. Router gradients flow from LM loss (STE)
7. Noisy routing affects routing probs while clean probs remain stable
8. Generation works (warm-start sanity)
9. Attention masking is preserved
10. Routing covers multiple experts for a large batch

Usage:
    python moe_emergence/verify_gpt2_integration.py
"""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from gpt2_moe import (
    MoEWrapper,
    collect_aux_outputs,
    compute_load_balance_loss,
    compute_z_loss,
    install_moe_layers,
)


def _set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _capture_mlp_io(
    model: GPT2LMHeadModel, inputs: dict, layer_indices: list[int]
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    inputs_by_layer: dict[int, torch.Tensor] = {}
    outputs_by_layer: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, inp, out):
            inputs_by_layer[layer_idx] = inp[0].detach().cpu()
            outputs_by_layer[layer_idx] = out.detach().cpu()

        return hook

    for layer_idx in layer_indices:
        mlp = model.transformer.h[layer_idx].mlp
        handles.append(mlp.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        _ = model(**inputs)

    for handle in handles:
        handle.remove()

    return inputs_by_layer, outputs_by_layer


def test_moe_installation(
    model: GPT2LMHeadModel,
    moe_modules: dict,
    moe_layers: list[int],
    n_experts: int,
    topk: int,
    base_total_params: int,
    base_mlp_params: dict[int, int],
    hidden_dim: int,
) -> None:
    print("\n[Test 1] MoE installation...")

    assert set(moe_modules.keys()) == set(moe_layers), (
        f"Expected MoE layers {moe_layers}, got {sorted(moe_modules.keys())}"
    )

    for idx in range(model.config.n_layer):
        mlp = model.transformer.h[idx].mlp
        if idx in moe_layers:
            assert isinstance(mlp, MoEWrapper), f"Layer {idx} is not MoEWrapper"
        else:
            assert not isinstance(mlp, MoEWrapper), (
                f"Layer {idx} unexpectedly MoEWrapper"
            )

    for layer_idx, moe in moe_modules.items():
        assert moe.n_experts == n_experts, f"Layer {layer_idx} n_experts mismatch"
        assert moe.topk == topk, f"Layer {layer_idx} topk mismatch"

        moe_param_count = sum(p.numel() for p in moe.parameters())
        expected_moe_params = (
            n_experts * base_mlp_params[layer_idx] + n_experts * hidden_dim
        )
        assert moe_param_count == expected_moe_params, (
            f"Layer {layer_idx} param mismatch: expected {expected_moe_params}, "
            f"got {moe_param_count}"
        )

        if n_experts > 1:
            p0 = next(moe.experts[0].parameters())
            p1 = next(moe.experts[1].parameters())
            assert p0.data_ptr() != p1.data_ptr(), "Experts share parameters"
            assert not torch.allclose(p0, p1), "Experts are identical (no perturbation)"

    expected_total = (
        base_total_params
        - sum(base_mlp_params.values())
        + sum(
            n_experts * base_mlp_params[layer_idx] + n_experts * hidden_dim
            for layer_idx in moe_layers
        )
    )
    actual_total = sum(p.numel() for p in model.parameters())
    assert actual_total == expected_total, (
        f"Model param count mismatch: expected {expected_total}, got {actual_total}"
    )

    print("  [OK] MoE layers installed with expected parameters")


def test_warm_start_parity(
    moe_modules: dict,
    baseline_inputs: dict[int, torch.Tensor],
    baseline_outputs: dict[int, torch.Tensor],
    device: torch.device,
    max_rel_err: float = 0.02,
) -> None:
    print("\n[Test 2] Warm-start parity...")

    for layer_idx, moe in moe_modules.items():
        assert layer_idx in baseline_inputs, (
            f"Missing baseline input for layer {layer_idx}"
        )
        assert layer_idx in baseline_outputs, (
            f"Missing baseline output for layer {layer_idx}"
        )

        moe.eval()
        hidden_in = baseline_inputs[layer_idx].to(device)
        base_out = baseline_outputs[layer_idx].to(device)

        with torch.no_grad():
            moe_out = moe(hidden_in)

        assert moe_out.shape == base_out.shape, (
            f"Layer {layer_idx} output shape mismatch: {moe_out.shape} vs {base_out.shape}"
        )
        rel_err = (moe_out - base_out).abs().mean() / (base_out.abs().mean() + 1e-9)
        assert rel_err < max_rel_err, (
            f"Layer {layer_idx} warm-start drift too large: rel_err={rel_err:.4f}"
        )

        aux = moe.last_aux
        assert aux is not None, f"Layer {layer_idx} missing aux outputs"
        if moe.topk == 1:
            assert torch.allclose(
                aux.topk_weights, torch.ones_like(aux.topk_weights)
            ), f"Layer {layer_idx} top-1 weights not hard 1.0"

        print(f"  [OK] Layer {layer_idx} parity rel_err={rel_err:.4f}")


def test_forward_pass(model: GPT2LMHeadModel, tokenizer, device: torch.device) -> None:
    print("\n[Test 3] Forward pass...")

    model.eval()
    inputs = tokenizer(
        "The quick brown fox jumps over the lazy dog", return_tensors="pt"
    )
    inputs = _to_device(inputs, device)

    with torch.no_grad():
        outputs = model(**inputs)

    expected_shape = (1, inputs["input_ids"].shape[1], model.config.vocab_size)
    assert outputs.logits.shape == expected_shape, (
        f"Expected {expected_shape}, got {outputs.logits.shape}"
    )
    print("  [OK] Output shape correct")


def test_aux_outputs(
    model: GPT2LMHeadModel,
    tokenizer,
    moe_modules: dict,
    device: torch.device,
) -> dict[int, dict]:
    print("\n[Test 4] Auxiliary outputs...")

    model.eval()
    inputs = tokenizer("Hello world", return_tensors="pt")
    inputs = _to_device(inputs, device)

    with torch.no_grad():
        _ = model(**inputs)

    aux_outputs = collect_aux_outputs(moe_modules)
    aux_by_layer = {aux["layer_idx"]: aux for aux in aux_outputs}

    assert set(aux_by_layer.keys()) == set(moe_modules.keys()), (
        "Aux outputs missing for some MoE layers"
    )

    for layer_idx, moe in moe_modules.items():
        aux = aux_by_layer[layer_idx]
        n_tokens = aux["router_probs"].shape[0]
        n_experts = moe.n_experts
        topk = moe.topk

        assert aux["router_probs"].shape == (n_tokens, n_experts)
        assert aux["router_probs_clean"].shape == (n_tokens, n_experts)
        assert aux["router_logits"].shape == (n_tokens, n_experts)
        assert aux["topk_indices"].shape == (n_tokens, topk)
        assert aux["topk_weights"].shape == (n_tokens, topk)
        assert aux["entropy"].shape == (n_tokens,)

        clean_expected = torch.softmax(aux["router_logits"], dim=-1)
        assert torch.allclose(
            aux["router_probs_clean"], clean_expected, atol=1e-6, rtol=1e-5
        ), f"Layer {layer_idx} clean probs mismatch"
        assert torch.allclose(
            aux["router_probs"], aux["router_probs_clean"], atol=1e-6, rtol=1e-5
        ), f"Layer {layer_idx} router_probs != clean probs in eval"

        entropy_expected = -(clean_expected * torch.log(clean_expected + 1e-9)).sum(
            dim=-1
        )
        assert torch.allclose(aux["entropy"], entropy_expected, atol=1e-6, rtol=1e-5), (
            f"Layer {layer_idx} entropy mismatch"
        )

        probs_sum = aux["router_probs"].sum(dim=-1)
        clean_sum = aux["router_probs_clean"].sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-6), (
            f"Layer {layer_idx} router_probs do not sum to 1"
        )
        assert torch.allclose(clean_sum, torch.ones_like(clean_sum), atol=1e-6), (
            f"Layer {layer_idx} router_probs_clean do not sum to 1"
        )

        expected_indices = torch.topk(aux["router_probs"], k=topk, dim=-1).indices
        assert torch.equal(aux["topk_indices"], expected_indices), (
            f"Layer {layer_idx} topk indices mismatch"
        )

        assert aux["topk_indices"].min() >= 0
        assert aux["topk_indices"].max() < n_experts

        weights_sum = aux["topk_weights"].sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6), (
            f"Layer {layer_idx} topk weights do not sum to 1"
        )

        if topk == 1:
            assert torch.allclose(
                aux["topk_weights"], torch.ones_like(aux["topk_weights"])
            ), f"Layer {layer_idx} top-1 weights not hard 1.0"

        print(f"  [OK] Layer {layer_idx} aux outputs consistent")

    return aux_by_layer


def test_loss_computation(aux_by_layer: dict[int, dict], n_experts: int) -> None:
    print("\n[Test 5] Loss computation...")

    for layer_idx, aux in aux_by_layer.items():
        lb_loss = compute_load_balance_loss(
            aux["router_probs"], aux["topk_indices"], n_experts
        )
        z_loss = compute_z_loss(aux["router_logits"])

        assert torch.isfinite(lb_loss), f"LB loss not finite at layer {layer_idx}"
        assert torch.isfinite(z_loss), f"Z-loss not finite at layer {layer_idx}"
        assert lb_loss >= 1.0 - 1e-3, f"LB loss below minimum at layer {layer_idx}"
        assert lb_loss <= n_experts + 1e-3, f"LB loss above max at layer {layer_idx}"
        assert z_loss >= 0, f"Z-loss negative at layer {layer_idx}"

        print(
            f"  [OK] Layer {layer_idx} LB={lb_loss.item():.4f}, Z={z_loss.item():.4f}"
        )


def test_router_gradients(
    model: GPT2LMHeadModel, tokenizer, moe_modules: dict, device: torch.device
) -> None:
    print("\n[Test 6] Router gradient flow (STE)...")

    model.train()
    model.zero_grad()

    inputs = tokenizer("Test backward pass for router gradients", return_tensors="pt")
    inputs = _to_device(inputs, device)

    outputs = model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()

    for layer_idx, moe in moe_modules.items():
        router_grad = moe.router.gate.weight.grad
        assert router_grad is not None, f"Router at layer {layer_idx} has no gradient"
        assert torch.isfinite(router_grad).all(), (
            f"Router grad not finite at layer {layer_idx}"
        )
        grad_norm = router_grad.abs().sum().item()
        assert grad_norm > 0, f"Router gradient zero at layer {layer_idx}"
        print(f"  [OK] Layer {layer_idx} router grad norm = {grad_norm:.6f}")


def test_noisy_routing(
    model: GPT2LMHeadModel, tokenizer, moe_modules: dict, device: torch.device
) -> None:
    print("\n[Test 7] Noisy routing separation...")

    model.eval()
    inputs = tokenizer("Testing noisy routing behavior", return_tensors="pt")
    inputs = _to_device(inputs, device)

    for target_layer, moe in moe_modules.items():
        for other in moe_modules.values():
            other.router.eval()
        moe.router.train()
        moe.router.set_noise_annealing(total_steps=10, anneal_fraction=1.0)
        moe.router.training_step.zero_()

        with torch.no_grad():
            _ = model(**inputs)
            aux_first = collect_aux_outputs(moe_modules)
            _ = model(**inputs)
            aux_second = collect_aux_outputs(moe_modules)

        aux_first_map = {aux["layer_idx"]: aux for aux in aux_first}
        aux_second_map = {aux["layer_idx"]: aux for aux in aux_second}

        aux = aux_first_map[target_layer]
        aux_b = aux_second_map[target_layer]

        clean_expected = torch.softmax(aux["router_logits"], dim=-1)
        assert torch.allclose(
            aux["router_probs_clean"], clean_expected, atol=1e-6, rtol=1e-5
        ), f"Layer {target_layer} clean probs mismatch under noise"

        assert torch.allclose(
            aux["router_logits"], aux_b["router_logits"], atol=1e-6, rtol=1e-5
        ), f"Layer {target_layer} router_logits changed between noisy passes"
        assert torch.allclose(
            aux["router_probs_clean"],
            aux_b["router_probs_clean"],
            atol=1e-6,
            rtol=1e-5,
        ), f"Layer {target_layer} clean probs changed between noisy passes"

        diff_noisy = (aux["router_probs"] - aux_b["router_probs"]).abs().mean().item()
        diff_clean = (
            (aux["router_probs"] - aux["router_probs_clean"]).abs().mean().item()
        )

        assert diff_noisy > 1e-6, (
            f"Layer {target_layer} router_probs identical across noisy passes "
            f"(diff={diff_noisy:.2e})"
        )
        assert diff_clean > 1e-6, (
            f"Layer {target_layer} noisy routing not affecting probs "
            f"(diff={diff_clean:.2e})"
        )
        print(
            f"  [OK] Layer {target_layer} noisy Δ={diff_noisy:.2e}, "
            f"|noisy-clean|={diff_clean:.2e}"
        )


def test_generation(model: GPT2LMHeadModel, tokenizer, device: torch.device) -> None:
    print("\n[Test 8] Generation...")

    model.eval()
    inputs = tokenizer("The meaning of life is", return_tensors="pt")
    inputs = _to_device(inputs, device)

    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    assert generated.shape[1] > inputs["input_ids"].shape[1], (
        "Generation produced no new tokens"
    )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  [OK] Generated: {decoded}")


def test_attention_masking(
    model: GPT2LMHeadModel, tokenizer, device: torch.device
) -> None:
    print("\n[Test 9] Attention masking...")

    model.eval()
    texts = ["Short", "This is a much longer sentence that will require padding"]
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    batch = _to_device(batch, device)
    single = tokenizer(texts[0], return_tensors="pt")
    single = _to_device(single, device)

    with torch.no_grad():
        batch_outputs = model(**batch)
        single_outputs = model(**single)

    seq_len = single["input_ids"].shape[1]
    diff = (
        (batch_outputs.logits[0, :seq_len] - single_outputs.logits[0])
        .abs()
        .max()
        .item()
    )

    assert diff < 1e-4, f"Attention masking mismatch: max diff = {diff:.2e}"
    print(f"  [OK] Attention masking preserved (max diff = {diff:.2e})")


def test_routing_coverage(
    model: GPT2LMHeadModel, moe_modules: dict, device: torch.device
) -> None:
    print("\n[Test 10] Routing coverage...")

    model.eval()
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 512), device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    aux_outputs = collect_aux_outputs(moe_modules)
    for aux in aux_outputs:
        n_experts = aux["router_probs"].shape[1]
        indices = aux["topk_indices"].view(-1)
        counts = torch.bincount(indices, minlength=n_experts)
        assert counts.sum().item() == indices.numel()
        assert (counts > 0).all(), (
            f"Layer {aux['layer_idx']} has unused experts in routing coverage"
        )
        print(f"  [OK] Layer {aux['layer_idx']} routes to all experts")


def main() -> None:
    print("=" * 60)
    print("GPT-2 MoE Integration Verification")
    print("=" * 60)

    _set_seed(0)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    print("\nLoading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    moe_layers = [8, 9, 10, 11]
    n_experts = 8
    topk = 1
    noise_std = 0.1

    base_total_params = sum(p.numel() for p in model.parameters())
    base_mlp_params = {
        idx: sum(p.numel() for p in model.transformer.h[idx].mlp.parameters())
        for idx in moe_layers
    }

    print("\nCapturing baseline MLP I/O for parity check...")
    model.eval()
    parity_inputs = tokenizer(
        "Warm-start parity check input for GPT-2 MoE integration.",
        return_tensors="pt",
    )
    baseline_inputs, baseline_outputs = _capture_mlp_io(
        model, parity_inputs, moe_layers
    )

    print("\nInstalling MoE layers...")
    model, moe_modules = install_moe_layers(
        model,
        moe_layers=moe_layers,
        n_experts=n_experts,
        topk=topk,
        noise_std=noise_std,
    )

    test_moe_installation(
        model,
        moe_modules,
        moe_layers,
        n_experts,
        topk,
        base_total_params,
        base_mlp_params,
        model.config.n_embd,
    )

    test_warm_start_parity(
        moe_modules, baseline_inputs, baseline_outputs, torch.device("cpu")
    )

    model.to(device)
    test_forward_pass(model, tokenizer, device)
    aux_by_layer = test_aux_outputs(model, tokenizer, moe_modules, device)
    test_loss_computation(aux_by_layer, n_experts)
    test_router_gradients(model, tokenizer, moe_modules, device)
    test_noisy_routing(model, tokenizer, moe_modules, device)
    test_generation(model, tokenizer, device)
    test_attention_masking(model, tokenizer, device)
    test_routing_coverage(model, moe_modules, device)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nPhase 2 verification complete. Ready for Phase 3 (Dataset Preparation).")


if __name__ == "__main__":
    main()
