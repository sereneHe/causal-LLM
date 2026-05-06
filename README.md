# causal-llm

Unified preprocessing and baseline runners for causal discovery experiments.

## Azure workshop

This repository now includes an Azure Developer CLI (azd) workshop scaffold for provisioning shared cloud infrastructure.

- azd project file: `azure.yaml`
- Infrastructure as code: `infra/main.bicep`
- Parameter defaults: `infra/main.parameters.json`
- End-to-end deployment guide: `AZURE_WORKSHOP.md`

To get started quickly:

```bash
azd env new causal-llm-dev
azd env set AZURE_LOCATION eastus2
azd provision --preview
azd provision
azd deploy runner
```

## Secrets (Keychain / direnv / .env)

This repo is set up to load credentials via `direnv` from either a local `.env` (gitignored) or macOS Keychain.

- Create `.env` from `.env.example` if you want file-based local config.
- Recommended (macOS): store the key in Keychain and let `.envrc` export it:
	- `security add-generic-password -a "$USER" -s "anthropic_foundry_api_key" -w "<YOUR_KEY>" -U`
	- `cd /Users/xiaoyuhe/Causal-LLM && direnv allow`

Connectivity check (does not print the key):

- `python3 scripts/utils/check_anthropic_foundry.py --max-tokens 8`

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
