terraform {
  required_version = ">= 1.10, < 2.0"

  backend "s3" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = local.tags
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  name = var.project
  qualipilot_checks = merge(
    {
      missing_values                   = true
      duplicates                       = true
      data_types                       = true
      outliers                         = true
      ranges                           = true
      cardinality                      = true
      freshness                        = false
      min_rows                         = 1
      required_columns                 = []
      expected_dtypes                  = {}
      outlier_iqr_multiplier           = 1.5
      duplicate_subset                 = null
      column_ranges                    = {}
      freshness_columns                = []
      freshness_max_age_hours          = 24
      freshness_timezone               = "UTC"
      freshness_future_tolerance_hours = 0
      sample_size                      = 0
      include_top_values               = false
    },
    try(var.qualipilot_config.checks == null ? {} : var.qualipilot_config.checks, {}),
  )
  qualipilot_llm = merge(
    {
      provider        = "none"
      model           = ""
      max_tokens      = 1500
      temperature     = 0.2
      timeout_seconds = 60
      retries         = 3
      system_prompt   = "You are a senior data engineer. Given a data quality summary, produce a concise markdown report with findings, impact, and recommended cleanup steps. Be specific; avoid filler."
    },
    try(var.qualipilot_config.llm == null ? {} : var.qualipilot_config.llm, {}),
  )
  qualipilot_config = {
    engine = try(var.qualipilot_config.engine, "auto")
    checks = local.qualipilot_checks
    llm    = local.qualipilot_llm
  }
  configured_llm_model    = local.qualipilot_config.llm.model
  configured_llm_provider = local.qualipilot_config.llm.provider
  tags = {
    Project   = var.project
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------------------------------
# Storage
# -----------------------------------------------------------------------------

resource "aws_s3_bucket" "data" {
  bucket        = "${local.name}-${data.aws_caller_identity.current.account_id}-${var.region}"
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }

}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_iam_policy_document" "bucket_transport" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"
    actions = [
      "s3:*",
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "data" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.bucket_transport.json
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  depends_on = [aws_s3_bucket_versioning.data]

  rule {
    id     = "storage-hygiene"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  rule {
    id     = "expire-reports"
    status = "Enabled"

    filter {
      prefix = "reports/"
    }

    expiration {
      days = var.report_retention_days
    }
  }
}

# -----------------------------------------------------------------------------
# Logging and failed asynchronous invocations
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = var.log_retention_days
}

resource "aws_sqs_queue" "failures" {
  name                      = "${local.name}-failed-invocations"
  message_retention_seconds = 1209600
  kms_master_key_id         = "alias/aws/sqs"
}

# -----------------------------------------------------------------------------
# Lambda execution role
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
}

data "aws_iam_policy_document" "permissions" {
  statement {
    sid       = "ReadInputs"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.data.arn}/${var.input_prefix}*"]
  }

  statement {
    sid       = "WriteReports"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/reports/*"]

    condition {
      test     = "StringEquals"
      variable = "s3:x-amz-server-side-encryption"
      values   = ["AES256"]
    }
  }

  statement {
    sid       = "ReadReports"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/reports/*"]
  }

  statement {
    sid       = "WriteLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.lambda.arn}:*"]
  }

  statement {
    sid       = "WriteFailedInvocations"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.failures.arn]
  }

  dynamic "statement" {
    for_each = length(var.bedrock_model_arns) == 0 ? [] : [1]

    content {
      sid       = "InvokeApprovedBedrockModels"
      actions   = ["bedrock:InvokeModel"]
      resources = var.bedrock_model_arns
    }
  }

  dynamic "statement" {
    for_each = length(var.bedrock_inference_profiles) == 0 ? [] : [1]

    content {
      sid       = "InvokeApprovedBedrockProfiles"
      actions   = ["bedrock:InvokeModel"]
      resources = keys(var.bedrock_inference_profiles)
    }
  }

  dynamic "statement" {
    for_each = var.bedrock_inference_profiles

    content {
      actions   = ["bedrock:InvokeModel"]
      resources = statement.value

      condition {
        test     = "ArnEquals"
        variable = "bedrock:InferenceProfileArn"
        values   = [statement.key]
      }
    }
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${local.name}-lambda"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.permissions.json
}

# -----------------------------------------------------------------------------
# Immutable container image and Lambda
# -----------------------------------------------------------------------------

resource "aws_ecr_repository" "image" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE"

  lifecycle {
    precondition {
      condition     = length(base64encode(jsonencode(local.qualipilot_config))) <= 3600
      error_message = "The encoded qualipilot_config is too large for the Lambda environment."
    }
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "image" {
  repository = aws_ecr_repository.image.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after seven days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      }
    ]
  })
}

data "aws_iam_policy_document" "ecr_lambda" {
  statement {
    sid = "LambdaImageRead"

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }

    actions = [
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values = [
        "arn:${data.aws_partition.current.partition}:lambda:${var.region}:${data.aws_caller_identity.current.account_id}:function:${local.name}",
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_ecr_repository_policy" "lambda" {
  repository = aws_ecr_repository.image.name
  policy     = data.aws_iam_policy_document.ecr_lambda.json
}

resource "aws_lambda_function" "checker" {
  count = var.image_digest == null ? 0 : 1

  function_name                  = local.name
  role                           = aws_iam_role.lambda.arn
  package_type                   = "Image"
  image_uri                      = "${aws_ecr_repository.image.repository_url}@${coalesce(var.image_digest, "sha256:0000000000000000000000000000000000000000000000000000000000000000")}"
  timeout                        = var.timeout_seconds
  memory_size                    = var.memory_mb
  reserved_concurrent_executions = var.reserved_concurrency
  architectures                  = ["x86_64"]

  ephemeral_storage {
    size = var.ephemeral_storage_mb
  }

  environment {
    variables = {
      QUALIPILOT_JSON_LOGS         = "1"
      QUALIPILOT_LOG_LEVEL         = var.log_level
      QUALIPILOT_MAX_INPUT_BYTES   = tostring(var.max_input_bytes)
      QUALIPILOT_MAX_DATASET_BYTES = tostring(floor(var.memory_mb * 1024 * 1024 / 3))
      QUALIPILOT_FAIL_ON           = var.fail_on
      QUALIPILOT_CONFIG_JSON       = jsonencode(local.qualipilot_config)
      QUALIPILOT_BUILD_ID          = coalesce(var.image_digest, "unconfigured")
    }
  }

  lifecycle {
    precondition {
      condition     = length(var.alarm_action_arns) > 0
      error_message = "alarm_action_arns must configure production alerting."
    }

    precondition {
      condition = (
        local.configured_llm_provider != "bedrock"
        || var.timeout_seconds >= 120
      )
      error_message = "Bedrock reporting requires timeout_seconds of at least 120."
    }

    precondition {
      condition = (
        local.configured_llm_provider != "bedrock"
        || contains(
          concat(
            var.bedrock_model_arns,
            keys(var.bedrock_inference_profiles),
          ),
          local.configured_llm_model,
        )
      )
      error_message = "The configured Bedrock model must match an allowed model or inference-profile ARN."
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_ecr_repository_policy.lambda,
    aws_iam_role_policy.lambda,
  ]
}

resource "aws_lambda_function_event_invoke_config" "checker" {
  count = var.image_digest == null ? 0 : 1

  function_name                = aws_lambda_function.checker[0].function_name
  maximum_event_age_in_seconds = 21600
  maximum_retry_attempts       = 0

  destination_config {
    on_failure {
      destination = aws_sqs_queue.failures.arn
    }
  }
}

resource "aws_lambda_permission" "s3" {
  count = var.image_digest == null ? 0 : 1

  statement_id   = "AllowS3Invoke"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.checker[0].function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.data.arn
  source_account = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket_notification" "inputs" {
  count = var.image_digest == null ? 0 : 1

  bucket = aws_s3_bucket.data.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.checker[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = var.input_prefix
  }

  depends_on = [aws_lambda_permission.s3]
}

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-lambda-errors"
  alarm_description   = "The qualipilot Lambda returned an error."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    FunctionName = aws_lambda_function.checker[0].function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-lambda-throttles"
  alarm_description   = "The qualipilot Lambda was throttled."
  namespace           = "AWS/Lambda"
  metric_name         = "Throttles"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    FunctionName = aws_lambda_function.checker[0].function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "lambda_duration" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-lambda-duration"
  alarm_description   = "The qualipilot Lambda used at least 80% of its timeout."
  namespace           = "AWS/Lambda"
  metric_name         = "Duration"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = var.timeout_seconds * 1000 * 0.8
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    FunctionName = aws_lambda_function.checker[0].function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "lambda_concurrency" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-lambda-concurrency"
  alarm_description   = "The qualipilot Lambda approached its concurrency limit."
  namespace           = "AWS/Lambda"
  metric_name         = "ConcurrentExecutions"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = max(1, floor(var.reserved_concurrency * 0.8))
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    FunctionName = aws_lambda_function.checker[0].function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "destination_delivery_failures" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-destination-delivery-failures"
  alarm_description   = "Lambda could not deliver a failed invocation to SQS."
  namespace           = "AWS/Lambda"
  metric_name         = "DestinationDeliveryFailures"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    FunctionName = aws_lambda_function.checker[0].function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "quality_gate_failures" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-quality-gate-failures"
  alarm_description   = "A dataset crossed the configured quality threshold."
  namespace           = "Qualipilot"
  metric_name         = "QualityGateFailures"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    FunctionName = aws_lambda_function.checker[0].function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "llm_generation_failures" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-llm-generation-failures"
  alarm_description   = "Bedrock narrative generation failed."
  namespace           = "Qualipilot"
  metric_name         = "LLMGenerationFailures"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    FunctionName = aws_lambda_function.checker[0].function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "failure_queue_depth" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-failure-queue-depth"
  alarm_description   = "Failed Lambda invocations are waiting in SQS."
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    QueueName = aws_sqs_queue.failures.name
  }
}

resource "aws_cloudwatch_metric_alarm" "failure_queue_age" {
  count = var.image_digest == null ? 0 : 1

  alarm_name          = "${local.name}-failure-queue-age"
  alarm_description   = "A failed invocation has waited in SQS for 15 minutes."
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateAgeOfOldestMessage"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 900
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_action_arns
  ok_actions          = var.alarm_action_arns

  dimensions = {
    QueueName = aws_sqs_queue.failures.name
  }
}
