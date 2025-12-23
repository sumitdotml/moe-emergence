Document a training run or experiment. The user will describe what they ran.

Instructions:
1. Read the template at `docs/experiments/_TEMPLATE.md`
2. Gather the following information (ask if not provided):
   - Run objective (what question is this answering?)
   - Configuration used (model, training, loss coefficients)
   - Results (loss curves, metrics, observations)
   - Any anomalies or issues encountered
   - Cost (duration, GPU hours if known)
3. Determine the next run number by listing existing files in `docs/experiments/`
4. Create the experiment document following the template
5. Include a clear conclusion: what did this run teach us?
6. Get current git commit hash for reproducibility

If the user provides a wandb link or log file path, read it to extract metrics.

User's experiment to document: $ARGUMENTS