# NexusOmegaCore — Google Cloud Platform Infrastructure
# Usage: terraform init && terraform plan && terraform apply

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "db_password" {
  description = "Cloud SQL database password"
  type        = string
  sensitive   = true
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "nexus_vpc" {
  name                    = "nexus-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "nexus_subnet" {
  name          = "nexus-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.nexus_vpc.id
}

# Cloud NAT for outbound internet access
resource "google_compute_router" "nexus_router" {
  name    = "nexus-router"
  region  = var.region
  network = google_compute_network.nexus_vpc.id
}

resource "google_compute_router_nat" "nexus_nat" {
  name                               = "nexus-nat"
  router                             = google_compute_router.nexus_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# VPC Connector for Cloud Run
resource "google_vpc_access_connector" "nexus_connector" {
  name          = "nexus-vpc-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.nexus_vpc.name
}

# Cloud SQL PostgreSQL 16 with pgvector
resource "google_sql_database_instance" "nexus_db" {
  name             = "nexus-postgres"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier              = "db-custom-2-4096"
    availability_type = "ZONAL"

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.nexus_vpc.id
    }

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "nexus_database" {
  name     = "jarvis"
  instance = google_sql_database_instance.nexus_db.name
}

resource "google_sql_user" "nexus_user" {
  name     = "jarvis"
  instance = google_sql_database_instance.nexus_db.name
  password = var.db_password
}

# Memorystore Redis
resource "google_redis_instance" "nexus_redis" {
  name               = "nexus-redis"
  tier               = "BASIC"
  memory_size_gb     = 1
  region             = var.region
  authorized_network = google_compute_network.nexus_vpc.id
  redis_version      = "REDIS_7_0"
}

# Secret Manager secrets
locals {
  secrets = [
    "database-url",
    "redis-url",
    "telegram-bot-token",
    "jwt-secret-key",
    "gemini-api-key",
    "openai-api-key",
    "anthropic-api-key",
    "xai-api-key",
    "deepseek-api-key",
    "demo-unlock-code",
    "bootstrap-admin-code",
    "postgres-password",
  ]
}

resource "google_secret_manager_secret" "secrets" {
  for_each  = toset(local.secrets)
  secret_id = each.value

  replication {
    auto {}
  }
}

# Cloud Run Backend Service
resource "google_cloud_run_v2_service" "nexus_backend" {
  name     = "nexus-backend"
  location = var.region

  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    vpc_access {
      connector = google_vpc_access_connector.nexus_connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = "gcr.io/${var.project_id}/nexus-backend:latest"

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      startup_probe {
        http_get {
          path = "/api/v1/health"
          port = 8000
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 12
      }

      liveness_probe {
        http_get {
          path = "/api/v1/health"
          port = 8000
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }
}

# Cloud Run Bot Service
resource "google_cloud_run_v2_service" "nexus_bot" {
  name     = "nexus-telegram-bot"
  location = var.region

  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 3
    }

    vpc_access {
      connector = google_vpc_access_connector.nexus_connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = "gcr.io/${var.project_id}/nexus-telegram-bot:latest"

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }
  }
}

# Outputs
output "backend_url" {
  value = google_cloud_run_v2_service.nexus_backend.uri
}

output "bot_url" {
  value = google_cloud_run_v2_service.nexus_bot.uri
}

output "database_connection" {
  value     = google_sql_database_instance.nexus_db.connection_name
  sensitive = true
}

output "redis_host" {
  value = google_redis_instance.nexus_redis.host
}
