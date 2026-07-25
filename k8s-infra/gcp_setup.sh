#!/usr/bin/env bash
# AetherControl: Automated GCP GKE Zonal GPU Cluster Provisioning Script

set -e

# Configuration Variables
PROJECT_ID=${1:-"starry-trilogy-503219-s4"}
ZONE=${2:-"us-central1-a"}
CLUSTER_NAME="aethercontrol-gke"

echo "🚀 Setting up GCP Project: $PROJECT_ID in Zone: $ZONE..."

# 1. Set gcloud active project
gcloud config set project "$PROJECT_ID"

# 2. Enable necessary GCP APIs
echo "📦 Enabling Kubernetes Engine and Compute APIs..."
gcloud services enable container.googleapis.com compute.googleapis.com containerregistry.googleapis.com

# 3. Create Zonal GKE Cluster directly with Spot NVIDIA L4 GPU
echo "☸️ Creating Zonal GKE Cluster with Spot NVIDIA L4 GPU: $CLUSTER_NAME..."
gcloud container clusters create "$CLUSTER_NAME" \
    --zone "$ZONE" \
    --num-nodes 1 \
    --machine-type "g2-standard-4" \
    --accelerator type=nvidia-l4,count=1 \
    --spot \
    --scopes "https://www.googleapis.com/auth/cloud-platform"

# 4. Get Cluster Credentials for kubectl
echo "🔑 Fetching cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "$ZONE"

# 5. Apply NVIDIA GPU Driver DaemonSet
echo "🛠️ Installing NVIDIA GPU Drivers on GKE..."
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-drivers/master/gpu/nvidia-driver-installer/ubuntu/daemonset-unified.yaml

echo "✨ GCP GKE Zonal Cluster Setup Complete! Connected via kubectl."
