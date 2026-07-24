# Learning Dossier 05: GCP Infrastructure Architecture & GKE Deployment

> **Folder Path**: `docs/dossier/05_gcp_infrastructure_setup.md`  
> **Session Topic**: Google Cloud Platform (GCP) GKE Cluster Setup, Spot GPU Node Pools, and Cost-Optimized Deployment  
> **Date**: July 22, 2026  

---

## 1. GCP Architecture Choice: Proof of Concept

When initializing GCP infrastructure for AetherControl, select **Proof of Concept**.

### Why "Proof of Concept" Over Enterprise Production?
- **Speed & Simplicity**: Automatically provisions project resources and billing without forcing complex enterprise landing zones, shared VPCs, or corporate proxy routing.
- **Full Capabilities**: Grants complete administrative access to GKE, Compute Engine GPU VMs, and Cloud Storage.

---

## 2. Cost-Optimized GKE Node Pool Architecture

To minimize cloud spending while providing real NVIDIA GPU hardware for vLLM:

```
+-----------------------------------------------------------------------------------+
|                              GCP GKE CLUSTER                                      |
|                                                                                   |
|  [ CPU Node Pool: n1-standard-4 ] ──► Runs Prometheus, Grafana, & Control Plane  |
|                                                                                   |
|  [ GPU Node Pool (SPOT): g2-standard-4 + NVIDIA L4 ]                              |
|         └──► Runs vLLM Container + PagedAttention + trainsight InitContainer     |
|         └──► Saves 60-70% on GPU hourly costs via Spot VMs                        |
+-----------------------------------------------------------------------------------+
```

---

## 3. Step-by-Step Execution Commands

### Step A: Authenticate with Google Cloud CLI
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Step B: Run Provisioning Script
```bash
cd k8s-infra
./gcp_setup.sh YOUR_PROJECT_ID us-central1
```

### Step C: Deploy AetherControl Stack
```bash
# 1. Deploy Prometheus & Grafana monitoring
kubectl apply -f manifests/prometheus-grafana.yaml

# 2. Deploy vLLM Serving Engine with trainsight InitContainer
kubectl apply -f manifests/vllm-deployment.yaml

# 3. Check deployment status
kubectl get pods -w
```
