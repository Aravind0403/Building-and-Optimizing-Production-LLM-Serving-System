#!/usr/bin/env bash
# AetherControl: Automated GCP GKE Cluster Provisioning Script

set -e

# Configuration Variables
PROJECT_ID=${1:-"aethercontrol-dev"}
REGION=${2:-"us-central1"}
CLUSTER_NAME="aethercontrol-gke"

echo "🚀 Setting up GCP Project: $PROJECT_ID in Region: $REGION..."

# 1. Set gcloud active project
gcloud config set project "$PROJECT_ID"

# 2. Enable necessary GCP APIs
echo "📦 Enabling Kubernetes Engine and Compute APIs..."
gcloud services enable container.googleapis.com compute.googleapis.com containerregistry.googleapis.com

# 3. Create GKE Cluster with GPU support
echo "☸️ Creating GKE Cluster: $CLUSTER_NAME..."
gcloud container clusters create "$CLUSTER_NAME" \
    --region "$REGION" \
    --num-nodes 1 \
    --machine-type "n1-standard-4" \
    --scopes "https://www.googleapis.com/auth/cloud-platform" \
    --enable-autoscaling --min-nodes 1 --max-nodes 3

# 4. Add GPU Node Pool (NVIDIA L4 / T4 for cost efficiency)
echo "🎮 Adding GPU Node Pool..."
gcloud container node-pools create "gpu-pool" \
    --cluster "$CLUSTER_NAME" \
    --region "$REGION" \
    --machine-type "g2-standard-4" \
    --accelerator type=nvidia-l4,count=1 \
    --num-nodes 1 \
    --spot \
    --enable-autoscaling --min-nodes 0 --max-nodes 2

# 5. Get Cluster Credentials for kubectl
echo "🔑 Fetching cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$REGION"

# 6. Apply NVIDIA GPU Driver DaemonSet
echo "🛠️ Installing NVIDIA GPU Drivers on GKE..."
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-drivers/master/gpu/nvidia-driver-installer/ubuntu/daemonset-unified.yaml

echo "✨ GCP GKE Cluster Setup Complete! You are now connected via kubectl."
