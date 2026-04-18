output "ecr_repository_url" {
  value       = aws_ecr_repository.image.repository_url
  description = "Push the container image here before `terraform apply`."
}

output "reports_bucket" {
  value       = aws_s3_bucket.reports.bucket
  description = "S3 bucket reports are written to."
}

output "lambda_function_arn" {
  value       = aws_lambda_function.checker.arn
  description = "Function ARN you can wire into an EventBridge rule."
}
