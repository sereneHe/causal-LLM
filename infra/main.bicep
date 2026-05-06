targetScope = 'resourceGroup'

@description('A short environment prefix used for resource naming.')
param namePrefix string

@description('Primary deployment location.')
param location string = resourceGroup().location

@description('Deploy an Azure Container Registry for future containerized workloads.')
param deployAcr bool = true

@description('Tags applied to all resources in this template.')
param tags object = {
  workload: 'causal-llm'
  environment: 'workshop'
  managedBy: 'azd'
}

var compactPrefix = toLower(replace(replace(namePrefix, '-', ''), '_', ''))
var resourceToken = toLower(uniqueString(subscription().subscriptionId, resourceGroup().id, compactPrefix))

var logAnalyticsName = take('law-${compactPrefix}-${take(resourceToken, 8)}', 63)
var userAssignedIdentityName = take('id-${compactPrefix}-${take(resourceToken, 8)}', 64)
var keyVaultName = take('kv-${compactPrefix}-${take(resourceToken, 8)}', 24)
var storageAccountName = take('st${compactPrefix}${take(resourceToken, 11)}', 24)
var containerRegistryName = take('cr${compactPrefix}${resourceToken}', 50)
var containerAppsEnvironmentName = take('cae-${compactPrefix}-${take(resourceToken, 8)}', 64)

module logAnalytics 'br/public:avm/res/operational-insights/workspace:0.15.0' = {
  name: 'logAnalyticsDeployment'
  params: {
    name: logAnalyticsName
    location: location
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    tags: tags
  }
}

module userAssignedIdentity 'br/public:avm/res/managed-identity/user-assigned-identity:0.5.0' = {
  name: 'userAssignedIdentityDeployment'
  params: {
    name: userAssignedIdentityName
    location: location
    tags: tags
  }
}

module keyVault 'br/public:avm/res/key-vault/vault:0.13.3' = {
  name: 'keyVaultDeployment'
  params: {
    name: keyVaultName
    location: location
    enablePurgeProtection: true
    enableRbacAuthorization: true
    softDeleteRetentionInDays: 90
    tags: tags
  }
}

module storage 'br/public:avm/res/storage/storage-account:0.32.0' = {
  name: 'storageDeployment'
  params: {
    name: storageAccountName
    location: location
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
    requireInfrastructureEncryption: true
    blobServices: {
      containers: [
        {
          name: 'datasets'
          publicAccess: 'None'
        }
        {
          name: 'reports'
          publicAccess: 'None'
        }
      ]
    }
    tags: tags
  }
}

module containerRegistry 'br/public:avm/res/container-registry/registry:0.11.0' = if (deployAcr) {
  name: 'containerRegistryDeployment'
  params: {
    name: containerRegistryName
    location: location
    acrSku: 'Standard'
    acrAdminUserEnabled: false
    anonymousPullEnabled: false
    tags: tags
  }
}

module containerAppsEnvironment 'br/public:avm/res/app/managed-environment:0.13.2' = {
  name: 'containerAppsEnvironmentDeployment'
  params: {
    name: containerAppsEnvironmentName
    location: location
    appLogsConfiguration: {
      destination: 'azure-monitor'
    }
    publicNetworkAccess: 'Enabled'
    zoneRedundant: false
    tags: tags
  }
}

module acrPullRoleAssignment 'br/public:avm/ptn/authorization/resource-role-assignment:0.1.2' = if (deployAcr) {
  name: 'acrPullRoleAssignmentDeployment'
  params: {
    principalId: userAssignedIdentity.outputs.principalId
    resourceId: containerRegistry.outputs.resourceId
    roleDefinitionId: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  }
}

output logAnalyticsWorkspaceName string = logAnalytics.outputs.name
output logAnalyticsWorkspaceResourceId string = logAnalytics.outputs.resourceId

output userAssignedIdentityName string = userAssignedIdentity.outputs.name
output userAssignedIdentityClientId string = userAssignedIdentity.outputs.clientId
output userAssignedIdentityPrincipalId string = userAssignedIdentity.outputs.principalId
output userAssignedIdentityResourceId string = userAssignedIdentity.outputs.resourceId

output keyVaultName string = keyVault.outputs.name
output keyVaultUri string = keyVault.outputs.uri
output keyVaultResourceId string = keyVault.outputs.resourceId

output storageAccountName string = storage.outputs.name
output storageAccountResourceId string = storage.outputs.resourceId
output storageBlobEndpoint string = storage.outputs.primaryBlobEndpoint

output containerAppsEnvironmentName string = containerAppsEnvironment.outputs.name
output containerAppsEnvironmentResourceId string = containerAppsEnvironment.outputs.resourceId

output containerRegistryName string = deployAcr ? containerRegistry.outputs.name : ''
output containerRegistryLoginServer string = deployAcr ? containerRegistry.outputs.loginServer : ''
output containerRegistryResourceId string = deployAcr ? containerRegistry.outputs.resourceId : ''

output containerRegistryEndpoint string = deployAcr ? containerRegistry.outputs.loginServer : ''

output serviceRunnerIdentityId string = userAssignedIdentity.outputs.resourceId
