#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error

#######################
# User-Configurable Variables
#######################

# Subscription and Resource Details
SUBSCRIPTION_ID="<YOUR_SUBSCRIPTION_ID>"
RESOURCE_GROUP="ml-resource-group"
LOCATION="eastus"

# Storage Account Details
STORAGE_ACCOUNT="mlstorage$(date +%s)"
CONTAINER_NAME="ml-input-data"

# Azure ML Workspace Details
WORKSPACE_NAME="ml-workspace"

# Compute Cluster Details
COMPUTE_CLUSTER_NAME="ml-compute-cluster"
VM_SIZE="STANDARD_DS3_V2"
MIN_INSTANCES=0
MAX_INSTANCES=4

# Data Upload Details
LOCAL_DATA_PATH="../job_data/"
DATA_FILE_PATTERN="*.swf"

#######################

check_azure_cli() {
    if ! command -v az &> /dev/null
    then
        echo "Azure CLI not found. Installing Azure CLI..."
        install_azure_cli
    else
        echo "Azure CLI is already installed."
    fi
}

install_azure_cli() {
    # Detect OS type
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew update && brew install azure-cli
    else
        echo "Unsupported OS. Please install Azure CLI manually."
        exit 1
    fi
}

azure_login() {
    echo "Logging into Azure..."
    az login
    # Optionally, set the subscription if not the default
    if [ "$SUBSCRIPTION_ID" != "<YOUR_SUBSCRIPTION_ID>" ]; then
        echo "Setting Azure subscription to $SUBSCRIPTION_ID"
        az account set --subscription "$SUBSCRIPTION_ID"
    fi
}

create_resource_group() {
    echo "Creating Resource Group: $RESOURCE_GROUP in $LOCATION..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
}

create_storage_account() {
    echo "Creating Storage Account: $STORAGE_ACCOUNT..."
    az storage account create \
        --name "$STORAGE_ACCOUNT" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        --kind StorageV2

    STORAGE_KEY=$(az storage account keys list \
        --resource-group "$RESOURCE_GROUP" \
        --account-name "$STORAGE_ACCOUNT" \
        --query "[0].value" -o tsv)

    echo "Creating Blob Container: $CONTAINER_NAME..."
    az storage container create \
        --name "$CONTAINER_NAME" \
        --account-name "$STORAGE_ACCOUNT" \
        --account-key "$STORAGE_KEY" \
        --public-access off
}

create_ml_workspace() {
    echo "Creating Azure ML Workspace: $WORKSPACE_NAME..."
    az ml workspace create \
        --name "$WORKSPACE_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --storage-account "$STORAGE_ACCOUNT" \
        --container-name "$CONTAINER_NAME"
}

create_compute_cluster() {
    echo "Creating Compute Cluster: $COMPUTE_CLUSTER_NAME..."
    az ml compute create \
        --name "$COMPUTE_CLUSTER_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$WORKSPACE_NAME" \
        --type amlcompute \
        --min-instances "$MIN_INSTANCES" \
        --max-instances "$MAX_INSTANCES" \
        --vm-size "$VM_SIZE"
}

upload_data() {
    echo "Uploading data from $LOCAL_DATA_PATH to Blob Storage..."
    # Install Azure Storage Blob if not installed
    if ! pip show azure-storage-blob &> /dev/null
    then
        echo "Installing azure-storage-blob Python package..."
        pip install azure-storage-blob
    fi

    az storage blob upload-batch \
        --account-name "$STORAGE_ACCOUNT" \
        --account-key "$STORAGE_KEY" \
        --destination "$CONTAINER_NAME" \
        --source "$LOCAL_DATA_PATH" \
        --pattern "$DATA_FILE_PATTERN"

    echo "Data upload completed."
}

display_info() {
    echo "========================================"
    echo "Azure ML Environment Setup Completed!"
    echo "========================================"
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Storage Account: $STORAGE_ACCOUNT"
    echo "Blob Container: $CONTAINER_NAME"
    echo "Azure ML Workspace: $WORKSPACE_NAME"
    echo "Compute Cluster: $COMPUTE_CLUSTER_NAME"
    echo "Data Blob URI: https://$STORAGE_ACCOUNT.blob.core.windows.net/$CONTAINER_NAME/"
    echo "========================================"
    echo "Next Steps:"
    echo "1. Ensure your pipeline script is configured with the correct data URI:"
    echo "   -data https://$STORAGE_ACCOUNT.blob.core.windows.net/$CONTAINER_NAME/"
    echo "2. Run your pipeline script."
    echo "3. Monitor your pipeline via the Azure Portal or Azure ML SDK."
}

#######################

# Check if variables are set
if [[ "$SUBSCRIPTION_ID" == "<YOUR_SUBSCRIPTION_ID>" ]]; then
    echo "Please update the SUBSCRIPTION_ID variable in the script with your Azure Subscription ID."
    exit 1
fi

if [[ ! -d "$LOCAL_DATA_PATH" ]]; then
    echo "Local data path $LOCAL_DATA_PATH does not exist. Please update the LOCAL_DATA_PATH variable."
    exit 1
fi

#######################

# Step 1: Check and Install Azure CLI
check_azure_cli
# Step 2: Log into Azure
azure_login
# Step 3: Create Resource Group
create_resource_group
# Step 4: Create Storage Account and Container
create_storage_account
# Step 5: Create Azure ML Workspace
create_ml_workspace
# Step 6: Create Compute Cluster
create_compute_cluster
# Step 7: Upload Data to Blob Storage
upload_data
# Step 8: Display Setup Information
display_info
