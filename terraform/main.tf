terraform {
  required_version = ">= 1.3.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. GCS Bucket for Versioned Datasets
resource "google_storage_bucket" "datasets_bucket" {
  name                        = "${var.project_id}-aethercontrol-datasets"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

# 2. Artifact Registry for Container Images
resource "google_artifact_registry_repository" "aethercontrol_repo" {
  location      = var.region
  repository_id = "aethercontrol-repo"
  description   = "Docker repository for AetherControl container images"
  format        = "DOCKER"
}

# 3. GKE Cluster (Standard/Autopilot setup)
resource "google_container_cluster" "primary" {
  name                     = "aethercontrol-gke"
  location                 = var.region
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false
}

# 4. Spot GPU Node Pool (NVIDIA L4 / 24GB VRAM) for 70% Cost Savings
resource "google_container_node_pool" "spot_gpu_pool" {
  name       = "spot-gpu-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    spot         = true  # Preemptible Spot instance for ~70% cost optimization
    machine_type = "g2-standard-4"

    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      env = "production-dev"
    }
  }
}
