# Azure Workshop for causal-llm

This workshop adds an Azure Developer CLI (azd) workflow to provision secure shared infrastructure for running causal discovery experiments.

## What gets provisioned

- Log Analytics Workspace (monitoring and diagnostics)
- User Assigned Managed Identity (for future workload identity)
- Key Vault (RBAC enabled, purge protection enabled)
- Storage Account (Azure AD auth only, no shared key auth)
  - Blob containers: `datasets`, `reports`
- Azure Container Registry (optional support for future containerized execution)

All resources are deployed via Azure Verified Modules (AVM) in [infra/main.bicep](infra/main.bicep).

## Prerequisites

- Azure CLI (`az`) installed and logged in
- Azure Developer CLI (`azd`) installed
- Permission to create resources in the target subscription/resource group

## Quick start

1. Authenticate:

   ```bash
   az login
   azd auth login
   ```

2. Initialize environment:

   ```bash
   azd env new causal-llm-dev
   azd env set AZURE_LOCATION eastus2
   ```

3. Validate and preview the deployment:

   ```bash
   azd provision --preview
   ```

4. Provision infrastructure:

   ```bash
   azd provision
   ```

5. Deploy the cloud runner service:

  ```bash
  azd deploy runner
  ```

## Useful azd commands

- Show environment variables:

  ```bash
  azd env get-values
  ```

- Re-run infra deployment:

  ```bash
  azd provision
  ```

- Re-run the runner deployment:

  ```bash
  azd deploy runner
  ```

- Delete workshop resources:

  ```bash
  azd down
  ```

## Map outputs to local workflow

After provisioning, use these outputs from azd to configure local execution:

- `keyVaultUri`
- `storageBlobEndpoint`
- `userAssignedIdentityClientId`
- `containerRegistryLoginServer` (if ACR is enabled)

Check outputs:

```bash
azd env get-values | rg "KEYVAULT|STORAGE|IDENTITY|CONTAINER"
```

## Notes

- This workshop now provisions the shared Azure infrastructure plus a Container Apps Job-backed runner service.
- The runner image is built from the repository Dockerfile and executes the bundled `asia` benchmark by default.
