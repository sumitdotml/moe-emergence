"""
GPT-2 Inference Playground

Quick script to play around with GPT-2 prompting and generation.
Supports vanilla GPT-2, untrained MoE, and trained MoE from checkpoints.

Usage:
    # Vanilla GPT-2
    python moe_emergence/gpt2_inference.py

    # With untrained MoE layers (Phase 2)
    python moe_emergence/gpt2_inference.py --moe

    # With trained MoE checkpoint (Phase 5+)
    python moe_emergence/gpt2_inference.py --checkpoint checkpoints/run-002-step-10000.pt

    # Custom prompt
    python moe_emergence/gpt2_inference.py --prompt "Once upon a time"

    # Longer generation
    python moe_emergence/gpt2_inference.py --max-tokens 100

    # Sampling (more creative)
    python moe_emergence/gpt2_inference.py --sample --temperature 0.8 --top-p 0.9
"""

import argparse

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def main():
    parser = argparse.ArgumentParser(description="GPT-2 Inference Playground")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling (0 = disabled)"
    )
    parser.add_argument(
        "--moe",
        action="store_true",
        help="Use MoE-enhanced GPT-2 (layers 8-11)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained MoE checkpoint (e.g., checkpoints/run-002-step-10000.pt)",
    )
    parser.add_argument(
        "--moe-layers",
        type=int,
        nargs="+",
        default=[8, 9, 10, 11],
        help="Which layers to replace with MoE",
    )
    parser.add_argument(
        "--n-experts", type=int, default=8, help="Number of experts per MoE layer"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("GPT-2 Inference Playground")
    print("=" * 60)
    print(f"Device: {device}")

    # model type determination
    if args.checkpoint:
        model_type = "MoE (trained checkpoint)"
    elif args.moe:
        model_type = "MoE (untrained/warm-start)"
    else:
        model_type = "GPT-2 (vanilla)"
    print(f"Model: {model_type}")

    print("\nLoading model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # handling MoE: either load checkpoint or install fresh
    moe_modules = None
    if args.checkpoint:
        from gpt2_moe import install_moe_layers

        # first install MoE architecture (needed for state_dict loading)
        print(f"Installing MoE architecture at {args.moe_layers}...")
        model, moe_modules = install_moe_layers(
            model,
            moe_layers=args.moe_layers,
            n_experts=args.n_experts,
            topk=1,
        )

        # loading trained weights
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from step {checkpoint.get('step', 'unknown')}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")

    elif args.moe:
        from gpt2_moe import install_moe_layers

        print(f"Installing MoE layers at {args.moe_layers}...")
        model, moe_modules = install_moe_layers(
            model,
            moe_layers=args.moe_layers,
            n_experts=args.n_experts,
            topk=1,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")

    model.to(device)
    model.eval()

    gen_config = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if args.sample:
        gen_config["temperature"] = args.temperature
        gen_config["top_p"] = args.top_p
        if args.top_k > 0:
            gen_config["top_k"] = args.top_k

    print("\n" + "=" * 60)
    print("Generation Config")
    print("=" * 60)
    print(f'Prompt: "{args.prompt}"')
    print(f"Max tokens: {args.max_tokens}")
    print(f"Sampling: {args.sample}")
    if args.sample:
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
        print(f"Top-k: {args.top_k if args.top_k > 0 else 'disabled'}")

    print("\n" + "=" * 60)
    print("Generating...")
    print("=" * 60)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n{generated_text}\n")
    print("=" * 60)

    # showing routing stats if MoE
    if args.moe or args.checkpoint:
        from gpt2_moe import collect_aux_outputs

        print("\nMoE Routing Stats (last generation)")
        print("=" * 60)

        # running one more forward pass to get aux outputs
        with torch.no_grad():
            _ = model(**inputs)
        aux_outputs = collect_aux_outputs(moe_modules)

        for aux in aux_outputs:
            layer_idx = aux["layer_idx"]
            indices = aux["topk_indices"].cpu()
            n_experts = aux["router_probs"].shape[1]

            # expert usage count
            counts = torch.bincount(indices.view(-1), minlength=n_experts).float()
            fractions = counts / counts.sum()

            print(f"\nLayer {layer_idx}:")
            print("  Expert usage: ", end="")
            for i, frac in enumerate(fractions):
                print(f"E{i}={frac:.1%} ", end="")
            print()

            # entropy
            avg_entropy = aux["entropy"].mean().item()
            max_entropy = torch.log(torch.tensor(float(n_experts))).item()
            print(f"  Avg entropy: {avg_entropy:.3f} / {max_entropy:.3f}")

        print()


if __name__ == "__main__":
    main()
