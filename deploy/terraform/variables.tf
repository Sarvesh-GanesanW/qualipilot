variable "project" {
  description = "Short name, used for every resource prefix."
  type        = string
  default     = "qualipilot"
}

variable "region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Local aws profile name. Leave empty in CI."
  type        = string
  default     = null
}

variable "image_tag" {
  description = "ECR image tag deployed to the lambda."
  type        = string
  default     = "latest"
}

variable "timeout_seconds" {
  description = "Lambda function timeout in seconds."
  type        = number
  default     = 300
}

variable "memory_mb" {
  description = "Lambda memory in MB; CPU scales with memory."
  type        = number
  default     = 2048
}

variable "force_destroy" {
  description = "Allow terraform destroy to wipe non-empty buckets."
  type        = bool
  default     = false
}
