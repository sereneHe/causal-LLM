# causal-llm

Unified preprocessing and baseline runners for causal discovery experiments.

## Secrets (Keychain / direnv / .env)

This repo is set up to load credentials via `direnv` from either a local `.env` (gitignored) or macOS Keychain.

- Create `.env` from `.env.example` if you want file-based local config.
- Recommended (macOS): store the key in Keychain and let `.envrc` export it:
	- `security add-generic-password -a "$USER" -s "anthropic_foundry_api_key" -w "<YOUR_KEY>" -U`
	- `cd /Users/xiaoyuhe/Causal-LLM && direnv allow`

Connectivity check (does not print the key):

- `python3 scripts/utils/check_anthropic_foundry.py --max-tokens 8`

Notes:
- `https://ai.azure.com/...` (portal UI) and `https://<resource>.services.ai.azure.com/api/projects/...` (project management API) are **not** model inference endpoints.
- For Azure inference, you typically need an endpoint ending in `/chat/completions` (OpenAI-compatible) or (if using Anthropic directly) `/v1/messages`.

## SmBFO variable mapping

The original SmBFO notebook uses the following physical-symbol labels, which
map to the standardized variable names used in this repo:

| Notebook label | Repo variable |
| --- | --- |
| `$I_{14}$` | `Alkali_Cations` |
| `$I_5$` | `Transition_Metal_Cations` |
| `a` | `Lattice_Parameter` |
| `c` | `Composition` |
| `α` | `Unit_Cell_Angle` |
| `Vol` | `Volume` |
| `$P_x$` | `In_Plane_Polarization` |
