targetScope = 'resourceGroup'

@description('The Container Apps environment name created by the main infrastructure deployment.')
param containerAppsEnvironmentName string

@description('The container registry login server used to pull the runner image.')
param containerRegistryLoginServer string

@description('The fully qualified image name for the runner service.')
param imageName string

@description('The user-assigned managed identity resource ID used to authenticate with the container registry.')
param userAssignedIdentityResourceId string

@description('Primary deployment location.')
param location string = resourceGroup().location

@description('Tags applied to all resources in this template.')
param tags object = {
  workload: 'causal-llm'
  environment: 'workshop'
  managedBy: 'azd'
}

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-10-02-preview' existing = {
  name: containerAppsEnvironmentName
}

module runnerJob 'br/public:avm/res/app/job:0.7.1' = {
  name: 'runnerJobDeployment'
  params: {
    containers: [
      {
        name: 'runner'
        image: imageName
        env: [
          {
            name: 'PROBLEM_NAME'
            value: 'asia'
          }
          {
            name: 'ENABLED_MODELS'
            value: 'PC'
          }
          {
            name: 'OUTPUT_DIR'
            value: '/workspace/reports'
          }
          {
            name: 'KEEP_ALIVE'
            value: 'false'
          }
        ]
        resources: {
          cpu: '0.5'
          memory: '1Gi'
        }
      }
    ]
    environmentResourceId: containerAppsEnvironment.id
    name: 'runner'
    triggerType: 'Manual'
    location: location
    managedIdentities: {
      userAssignedResourceIds: [
        userAssignedIdentityResourceId
      ]
    }
    registries: [
      {
        server: containerRegistryLoginServer
        identity: userAssignedIdentityResourceId
      }
    ]
    tags: tags
    manualTriggerConfig: {}
  }
}

output runnerName string = runnerJob.outputs.name
output runnerResourceId string = runnerJob.outputs.resourceId