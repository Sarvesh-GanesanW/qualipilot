output "ecr_repository_url" {
  value       = aws_ecr_repository.image.repository_url
  description = "Build and push the Lambda image to this repository."
}

output "data_bucket" {
  value       = aws_s3_bucket.data.bucket
  description = "Upload supported files below the configured input prefix."
}

output "lambda_function_arn" {
  value       = try(aws_lambda_function.checker[0].arn, null)
  description = "Function ARN; null until image_digest is supplied."
}

output "failed_invocations_queue_url" {
  value       = aws_sqs_queue.failures.url
  description = "SQS queue containing exhausted asynchronous invocations."
}
