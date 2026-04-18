# Production-grade Lambda + S3 + IAM scaffold for datapilot.
# terraform init && terraform apply -var project=datapilot

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile
}

locals {
  name = var.project
  tags = {
    project   = var.project
    managedBy = "terraform"
  }
}

resource "aws_s3_bucket" "reports" {
  bucket        = "${local.name}-${data.aws_caller_identity.current.account_id}"
  force_destroy = var.force_destroy
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "reports" {
  bucket = aws_s3_bucket.reports.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "reports" {
  bucket = aws_s3_bucket.reports.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "reports" {
  bucket                  = aws_s3_bucket.reports.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_caller_identity" "current" {}

# -----------------------------------------------------------------------------
# IAM role for lambda
# -----------------------------------------------------------------------------

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${local.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "permissions" {
  statement {
    sid = "BedrockInvoke"
    actions = [
      "bedrock:Converse",
      "bedrock:ConverseStream",
      "bedrock:InvokeModel",
    ]
    resources = ["*"]
  }

  statement {
    sid     = "S3Access"
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.reports.arn,
      "${aws_s3_bucket.reports.arn}/*",
    ]
  }

  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${local.name}-lambda"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.permissions.json
}

# -----------------------------------------------------------------------------
# ECR repo + lambda function (container image)
# -----------------------------------------------------------------------------

resource "aws_ecr_repository" "image" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_lambda_function" "checker" {
  function_name = local.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.image.repository_url}:${var.image_tag}"
  timeout       = var.timeout_seconds
  memory_size   = var.memory_mb
  architectures = ["x86_64"]

  environment {
    variables = {
      DATAPILOT_JSON_LOGS = "1"
      DATAPILOT_LOG_LEVEL = "INFO"
    }
  }

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = 14
  tags              = local.tags
}
