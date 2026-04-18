# Deploy

Three deployment paths ship in-repo. Pick the one that matches the
runtime you already own; the Python code is identical across all of
them.

## 1. Local workstation

```bash
./install.sh --bedrock   # or --ollama / --all / --dev
datapilot check data.csv --llm bedrock --model provider.model-3-5-haiku-20241022-v1:0
```

Credentials come from the standard AWS chain — env vars, shared
config, SSO, whatever your `boto3` already resolves. Use `--profile`
to pick a named profile explicitly.

## 2. Docker (local or on-prem)

```bash
# single image, bedrock extras baked in
docker build -f docker/Dockerfile -t datapilot:latest .
docker run --rm -v $PWD/data:/data -v $PWD/reports:/reports \
    datapilot:latest check /data/input.csv \
    --output /reports/report.html

# full stack with an ollama sidecar
docker compose -f docker/docker-compose.yml up --build
```

The compose stack:

* brings up `ollama` on `:11434`
* builds `datapilot:latest` with the bedrock extra
* wires ollama into the container at `http://ollama:11434`
* mounts the host `~/.aws` directory read-only so `--llm bedrock`
  works too

## 3. AWS Lambda (container image)

```bash
cd deploy/terraform
terraform init
terraform apply \
    -var project=datapilot \
    -var region=us-east-1 \
    -var aws_profile=sre-tea
```

`terraform apply` creates:

* an S3 bucket (`<project>-<account>`) with versioning and SSE-S3
* an ECR repo
* an IAM role with least-privilege Bedrock + S3 + Logs access
* a Lambda function wired to the ECR image (`:latest` by default)
* a 14-day CloudWatch log group

Push the image:

```bash
ECR_URL=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password | docker login --username AWS --password-stdin "${ECR_URL%/*}"

docker build -f ../../docker/Dockerfile.lambda -t datapilot-lambda:latest ../..
docker tag datapilot-lambda:latest "${ECR_URL}:latest"
docker push "${ECR_URL}:latest"

aws lambda update-function-code --function-name datapilot --image-uri "${ECR_URL}:latest"
```

Invoke:

```bash
aws lambda invoke \
    --function-name datapilot \
    --payload '{"s3_uri":"s3://my-bucket/orders.parquet"}' \
    response.json
```

The handler downloads the S3 object, runs the checker, and writes
`s3://my-bucket/reports/orders.quality.json`.

### EventBridge schedule (optional)

Wire Lambda into EventBridge to run on a cron and use the
`--fail-on` severity flag to drop a SNS/Slack message when the check
degrades. Example (cron daily 09:00 UTC):

```hcl
resource "aws_cloudwatch_event_rule" "daily" {
  name                = "datapilot-daily"
  schedule_expression = "cron(0 9 * * ? *)"
}

resource "aws_cloudwatch_event_target" "daily" {
  rule      = aws_cloudwatch_event_rule.daily.name
  target_id = "datapilot"
  arn       = aws_lambda_function.checker.arn
  input     = jsonencode({ s3_uri = "s3://my-bucket/orders.parquet" })
}

resource "aws_lambda_permission" "events" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.checker.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily.arn
}
```

## Observability

* Set `DATAPILOT_JSON_LOGS=1` — the logging module emits one JSON
  line per record, which CloudWatch Logs Insights parses natively.
* Bedrock usage is logged at INFO per request: model id, input /
  output / total tokens. Aggregate those in a metric filter for a
  cost dashboard.
* `report.config_hash` is a stable SHA-1 of the config — use it as
  a dedup key in Athena / Snowflake when storing multiple runs.

## Security checklist

| Concern | Mitigation |
|---|---|
| Secrets in env | use AWS Secrets Manager / Parameter Store; `boto3` picks them up automatically |
| Sensitive samples in reports | cap `sample_size`, post-process samples before sharing |
| PII in LLM prompts | run with `llm.provider = "none"` or swap in a redacting provider |
| Bucket public access | Terraform enforces `BlockPublicAcls` + `BlockPublicPolicy` |
| IAM sprawl | `deploy/iam-policy.json` is the minimal policy for custom roles |
