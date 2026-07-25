variable "project_id" {
  type        = string
  description = "GCP Project ID"
  default     = "aethercontrol-dev"
}

variable "region" {
  type        = string
  description = "GCP Region for GKE and resources"
  default     = "us-central1"
}
