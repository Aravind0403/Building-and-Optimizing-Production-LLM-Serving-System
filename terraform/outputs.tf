output "kubernetes_cluster_name" {
  value       = google_container_cluster.primary.name
  description = "GKE Cluster Name"
}

output "gcs_bucket_name" {
  value       = google_storage_bucket.datasets_bucket.name
  description = "GCS Bucket for versioned datasets"
}

output "artifact_registry_url" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.aethercontrol_repo.repository_id}"
  description = "Artifact Registry Docker URL"
}
