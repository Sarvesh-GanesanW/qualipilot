variable "project" {
  description = "Short lowercase name used for resource names."
  type        = string
  default     = "qualipilot"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,31}$", var.project))
    error_message = "project must be 3-32 lowercase letters, digits, or hyphens."
  }
}

variable "region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}(-[a-z]+)+-[0-9]+$", var.region))
    error_message = "region must be an AWS region name such as us-east-1."
  }
}

variable "aws_profile" {
  description = "Local AWS profile. Leave null for CI or workload credentials."
  type        = string
  default     = null
  nullable    = true
}

variable "image_digest" {
  description = "Immutable ECR digest to deploy. Null creates the repository without Lambda."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition = (
      var.image_digest == null
      || can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))
    )
    error_message = "image_digest must be null or a sha256 digest."
  }
}

variable "input_prefix" {
  description = "S3 prefix that triggers checks and that Lambda may read."
  type        = string
  default     = "incoming/"

  validation {
    condition = (
      length(var.input_prefix) > 0
      && !startswith(var.input_prefix, "/")
      && endswith(var.input_prefix, "/")
      && !strcontains(var.input_prefix, "..")
      && !startswith(var.input_prefix, "reports/")
      && !strcontains(var.input_prefix, "*")
      && !strcontains(var.input_prefix, "?")
    )
    error_message = "input_prefix must be a safe prefix ending in / and outside reports/."
  }
}

variable "timeout_seconds" {
  description = "Lambda function timeout in seconds."
  type        = number
  default     = 300

  validation {
    condition = (
      floor(var.timeout_seconds) == var.timeout_seconds
      && var.timeout_seconds >= 1
      && var.timeout_seconds <= 900
    )
    error_message = "timeout_seconds must be an integer between 1 and 900."
  }
}

variable "memory_mb" {
  description = "Lambda memory in MB; CPU scales with memory."
  type        = number
  default     = 2048

  validation {
    condition = (
      floor(var.memory_mb) == var.memory_mb
      && var.memory_mb >= 128
      && var.memory_mb <= 10240
    )
    error_message = "memory_mb must be an integer between 128 and 10240."
  }
}

variable "ephemeral_storage_mb" {
  description = "Lambda /tmp allocation in MB."
  type        = number
  default     = 1024

  validation {
    condition = (
      floor(var.ephemeral_storage_mb) == var.ephemeral_storage_mb
      && var.ephemeral_storage_mb >= 512
      && var.ephemeral_storage_mb <= 10240
    )
    error_message = "ephemeral_storage_mb must be an integer between 512 and 10240."
  }
}

variable "max_input_bytes" {
  description = "Largest S3 object Lambda will download."
  type        = number
  default     = 268435456

  validation {
    condition = (
      floor(var.max_input_bytes) == var.max_input_bytes
      && var.max_input_bytes >= 1
      && var.max_input_bytes <= 1073741824
      && var.max_input_bytes <= (var.ephemeral_storage_mb - 64) * 1024 * 1024
    )
    error_message = "max_input_bytes must be an integer between 1 byte and 1 GiB and leave 64 MiB free in /tmp."
  }
}

variable "reserved_concurrency" {
  description = "Maximum concurrent Lambda executions."
  type        = number
  default     = 5

  validation {
    condition = (
      floor(var.reserved_concurrency) == var.reserved_concurrency
      && var.reserved_concurrency >= 1
    )
    error_message = "reserved_concurrency must be a positive integer."
  }
}

variable "fail_on" {
  description = "Lowest severity that marks the quality-gate outcome failed."
  type        = string
  default     = "error"

  validation {
    condition     = contains(["none", "warn", "error"], var.fail_on)
    error_message = "fail_on must be none, warn, or error."
  }
}

variable "log_level" {
  description = "Application log level."
  type        = string
  default     = "INFO"

  validation {
    condition = contains(
      ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
      var.log_level,
    )
    error_message = "log_level must be a standard uppercase Python log level."
  }
}

variable "qualipilot_config" {
  description = "Lambda-safe Qualipilot configuration for native S3 events. Terraform applies its documented primitive coercions and ignores extra object attributes."
  type = object({
    engine = optional(string, "auto")
    checks = optional(object({
      missing_values         = optional(bool, true)
      duplicates             = optional(bool, true)
      data_types             = optional(bool, true)
      outliers               = optional(bool, true)
      ranges                 = optional(bool, true)
      cardinality            = optional(bool, true)
      freshness              = optional(bool, false)
      min_rows               = optional(number, 1)
      required_columns       = optional(list(string), [])
      expected_dtypes        = optional(map(string), {})
      outlier_iqr_multiplier = optional(number, 1.5)
      duplicate_subset       = optional(list(string))
      column_ranges = optional(map(object({
        min = number
        max = number
      })), {})
      freshness_columns                = optional(list(string), [])
      freshness_max_age_hours          = optional(number, 24)
      freshness_timezone               = optional(string, "UTC")
      freshness_future_tolerance_hours = optional(number, 0)
      sample_size                      = optional(number, 0)
      include_top_values               = optional(bool, false)
    }), {})
    llm = optional(object({
      provider        = optional(string, "none")
      model           = optional(string, "")
      max_tokens      = optional(number, 1500)
      temperature     = optional(number, 0.2)
      timeout_seconds = optional(number, 60)
      retries         = optional(number, 3)
      system_prompt   = optional(string, "You are a senior data engineer. Given a data quality summary, produce a concise markdown report with findings, impact, and recommended cleanup steps. Be specific; avoid filler.")
    }), {})
  })
  default  = {}
  nullable = false

  validation {
    condition     = contains(["auto", "polars"], var.qualipilot_config.engine)
    error_message = "qualipilot_config.engine must be auto or polars."
  }

  validation {
    condition = (
      floor(var.qualipilot_config.checks.min_rows)
      == var.qualipilot_config.checks.min_rows
      && var.qualipilot_config.checks.min_rows >= 0
    )
    error_message = "qualipilot_config.checks.min_rows must be a non-negative integer."
  }

  validation {
    condition = alltrue([
      for columns in [
        var.qualipilot_config.checks.required_columns,
        var.qualipilot_config.checks.freshness_columns,
        coalesce(var.qualipilot_config.checks.duplicate_subset, []),
        ] : (
        alltrue([for column in columns : trimspace(column) != ""])
        && length(columns) == length(toset([
          for column in columns : lower(trimspace(column))
        ]))
      )
    ])
    error_message = "Qualipilot column lists must contain unique, non-empty names after trimming and case normalization."
  }

  validation {
    condition = (
      alltrue([
        for column, dtype in var.qualipilot_config.checks.expected_dtypes :
        trimspace(column) != "" && trimspace(dtype) != ""
      ])
      && length(var.qualipilot_config.checks.expected_dtypes) == length(toset([
        for column in keys(var.qualipilot_config.checks.expected_dtypes) :
        lower(trimspace(column))
      ]))
    )
    error_message = "qualipilot_config.checks.expected_dtypes must use unique, non-empty column names and non-empty dtype names."
  }

  validation {
    condition     = var.qualipilot_config.checks.outlier_iqr_multiplier > 0
    error_message = "qualipilot_config.checks.outlier_iqr_multiplier must be positive."
  }

  validation {
    condition = (
      alltrue([
        for column, bounds in var.qualipilot_config.checks.column_ranges :
        trimspace(column) != "" && bounds.max >= bounds.min
      ])
      && length(var.qualipilot_config.checks.column_ranges) == length(toset([
        for column in keys(var.qualipilot_config.checks.column_ranges) :
        lower(trimspace(column))
      ]))
    )
    error_message = "Every column range must have a unique, non-empty column name and max greater than or equal to min."
  }

  validation {
    condition     = var.qualipilot_config.checks.freshness_max_age_hours > 0
    error_message = "qualipilot_config.checks.freshness_max_age_hours must be positive."
  }

  validation {
    condition     = var.qualipilot_config.checks.freshness_timezone == "UTC"
    error_message = "Lambda supports only UTC for qualipilot_config.checks.freshness_timezone."
  }

  validation {
    condition     = var.qualipilot_config.checks.freshness_future_tolerance_hours >= 0
    error_message = "qualipilot_config.checks.freshness_future_tolerance_hours must be non-negative."
  }

  validation {
    condition = (
      floor(var.qualipilot_config.checks.sample_size)
      == var.qualipilot_config.checks.sample_size
      && var.qualipilot_config.checks.sample_size >= 0
      && var.qualipilot_config.checks.sample_size <= 1000
    )
    error_message = "qualipilot_config.checks.sample_size must be an integer between 0 and 1000."
  }

  validation {
    condition     = contains(["none", "bedrock"], var.qualipilot_config.llm.provider)
    error_message = "qualipilot_config.llm.provider must be none or bedrock."
  }

  validation {
    condition = (
      var.qualipilot_config.llm.provider == "none"
      || trimspace(var.qualipilot_config.llm.model) != ""
    )
    error_message = "Bedrock requires a non-empty qualipilot_config.llm.model."
  }

  validation {
    condition = (
      floor(var.qualipilot_config.llm.max_tokens)
      == var.qualipilot_config.llm.max_tokens
      && var.qualipilot_config.llm.max_tokens > 0
      && var.qualipilot_config.llm.max_tokens <= 64000
    )
    error_message = "qualipilot_config.llm.max_tokens must be an integer between 1 and 64000."
  }

  validation {
    condition = (
      var.qualipilot_config.llm.temperature >= 0
      && var.qualipilot_config.llm.temperature <= 2
      && (
        var.qualipilot_config.llm.provider != "bedrock"
        || var.qualipilot_config.llm.temperature <= 1
      )
    )
    error_message = "qualipilot_config.llm.temperature must be between 0 and 2, and at most 1 for Bedrock."
  }

  validation {
    condition     = var.qualipilot_config.llm.timeout_seconds > 0
    error_message = "qualipilot_config.llm.timeout_seconds must be positive."
  }

  validation {
    condition = (
      floor(var.qualipilot_config.llm.retries)
      == var.qualipilot_config.llm.retries
      && var.qualipilot_config.llm.retries >= 0
      && var.qualipilot_config.llm.retries <= 10
    )
    error_message = "qualipilot_config.llm.retries must be an integer between 0 and 10."
  }
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days."
  type        = number
  default     = 30

  validation {
    condition = contains(
      [1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1096, 1827, 2192, 2557, 2922, 3288, 3653],
      var.log_retention_days,
    )
    error_message = "log_retention_days must be a CloudWatch-supported value."
  }
}

variable "report_retention_days" {
  description = "Days before current report objects expire."
  type        = number
  default     = 90

  validation {
    condition = (
      floor(var.report_retention_days) == var.report_retention_days
      && var.report_retention_days >= 1
    )
    error_message = "report_retention_days must be a positive integer."
  }
}

variable "bedrock_model_arns" {
  description = "Exact Bedrock foundation-model ARNs Lambda may invoke directly."
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for arn in var.bedrock_model_arns :
      can(regex("^arn:[^:*?]+:bedrock:[^:*?]+::foundation-model/[^*?]+$", arn))
    ])
    error_message = "Every bedrock_model_arns entry must be an exact foundation-model ARN."
  }
}

variable "bedrock_inference_profiles" {
  description = "Inference-profile ARNs mapped to every foundation-model ARN they can route to."
  type        = map(list(string))
  default     = {}

  validation {
    condition = alltrue([
      for profile_arn, model_arns in var.bedrock_inference_profiles :
      can(regex("^arn:[^:*?]+:bedrock:[^:*?]+:[^:*?]+:(inference-profile|application-inference-profile)/[^*?]+$", profile_arn))
      && length(model_arns) > 0
      && alltrue([
        for model_arn in model_arns :
        can(regex("^arn:[^:*?]+:bedrock:([^:*?]+)?::foundation-model/[^*?]+$", model_arn))
        && split(":", model_arn)[1] == split(":", profile_arn)[1]
      ])
    ])
    error_message = "Each exact inference-profile ARN must map to all exact foundation-model ARNs it can route to."
  }
}

variable "alarm_action_arns" {
  description = "SNS or incident-management ARNs notified by CloudWatch alarms."
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for arn in var.alarm_action_arns :
      can(regex("^arn:[^:*?]+:[^:*?]+:[^:*?]*:[^:*?]*:[^*?]+$", arn))
    ])
    error_message = "alarm_action_arns must contain exact ARNs without wildcards."
  }
}

variable "force_destroy" {
  description = "Allow terraform destroy to delete a non-empty data bucket."
  type        = bool
  default     = false
}
