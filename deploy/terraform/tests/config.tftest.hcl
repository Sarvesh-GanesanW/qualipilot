mock_provider "aws" {}

override_data {
  target = data.aws_caller_identity.current
  values = {
    account_id = "123456789012"
  }
}

override_data {
  target = data.aws_partition.current
  values = {
    partition = "aws"
  }
}

override_data {
  target = data.aws_iam_policy_document.assume
  values = {
    json = "{}"
  }
}

override_data {
  target = data.aws_iam_policy_document.bucket_transport
  values = {
    json = "{}"
  }
}

override_data {
  target = data.aws_iam_policy_document.ecr_lambda
  values = {
    json = "{}"
  }
}

override_data {
  target = data.aws_iam_policy_document.permissions
  values = {
    json = "{}"
  }
}

run "valid_defaults" {
  command = plan

  variables {
    alarm_action_arns = [
      "arn:aws:sns:us-east-1:123456789012:qualipilot-alerts",
    ]
    image_digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }

  assert {
    condition = (
      jsondecode(
        aws_lambda_function.checker[0].environment[0].variables["QUALIPILOT_CONFIG_JSON"]
      ).checks.min_rows == 1
      && jsondecode(
        aws_lambda_function.checker[0].environment[0].variables["QUALIPILOT_CONFIG_JSON"]
      ).checks.freshness_timezone == "UTC"
    )
    error_message = "The default typed configuration must reach Lambda as valid JSON."
  }
}

run "accept_global_inference_profile_model" {
  command = plan

  variables {
    bedrock_inference_profiles = {
      "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.example-model" = [
        "arn:aws:bedrock:::foundation-model/example.model",
      ]
    }
  }
}

run "reject_negative_min_rows" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        min_rows = -1
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_pandas_engine" {
  command = plan

  variables {
    qualipilot_config = {
      engine = "pandas"
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_non_utc_freshness_timezone" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        freshness_timezone = "America/New_York"
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_normalized_duplicate_columns" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        required_columns = ["id", " ID "]
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_normalized_duplicate_dtype_columns" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        expected_dtypes = {
          id     = "int64"
          " ID " = "int64"
        }
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_inverted_column_range" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        column_ranges = {
          amount = {
            min = 10
            max = 0
          }
        }
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_normalized_duplicate_range_columns" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        column_ranges = {
          amount = {
            min = 0
            max = 10
          }
          " AMOUNT " = {
            min = 0
            max = 10
          }
        }
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_unknown_llm_provider" {
  command = plan

  variables {
    qualipilot_config = {
      llm = {
        provider = "openai"
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_bedrock_without_model" {
  command = plan

  variables {
    qualipilot_config = {
      llm = {
        provider = "bedrock"
        model    = " "
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_oversized_config" {
  command = plan

  variables {
    qualipilot_config = {
      llm = {
        system_prompt = join("", [for index in range(1000) : "xxxx"])
      }
    }
  }

  expect_failures = [
    aws_ecr_repository.image,
  ]
}

run "reject_unapproved_bedrock_model" {
  command = plan

  variables {
    alarm_action_arns = [
      "arn:aws:sns:us-east-1:123456789012:qualipilot-alerts",
    ]
    image_digest    = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    timeout_seconds = 120
    qualipilot_config = {
      llm = {
        provider = "bedrock"
        model    = "arn:aws:bedrock:us-east-1::foundation-model/example.model"
      }
    }
  }

  expect_failures = [
    aws_lambda_function.checker,
  ]
}

run "reject_non_positive_outlier_multiplier" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        outlier_iqr_multiplier = 0
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_non_positive_freshness_age" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        freshness_max_age_hours = 0
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_negative_future_tolerance" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        freshness_future_tolerance_hours = -1
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_oversized_sample" {
  command = plan

  variables {
    qualipilot_config = {
      checks = {
        sample_size = 1001
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_fractional_max_tokens" {
  command = plan

  variables {
    qualipilot_config = {
      llm = {
        max_tokens = 1.5
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_bedrock_temperature_above_one" {
  command = plan

  variables {
    qualipilot_config = {
      llm = {
        provider    = "bedrock"
        model       = "example.model"
        temperature = 1.1
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_non_positive_llm_timeout" {
  command = plan

  variables {
    qualipilot_config = {
      llm = {
        timeout_seconds = 0
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}

run "reject_excessive_llm_retries" {
  command = plan

  variables {
    qualipilot_config = {
      llm = {
        retries = 11
      }
    }
  }

  expect_failures = [
    var.qualipilot_config,
  ]
}
