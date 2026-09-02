# Deployment

This repository supports a local virtual environment, a batch container, a
local Ollama compose stack, and an S3-triggered AWS Lambda container.

## Local environment

From a source checkout:

```bash
./install.sh --bedrock
source .venv/bin/activate

test -n "$BEDROCK_MODEL_ID"
qualipilot check data.csv \
  --llm bedrock \
  --model "$BEDROCK_MODEL_ID" \
  --region us-east-1
```

On Windows, use `.\install.ps1 -Extras bedrock` and activate
`.venv\Scripts\Activate.ps1`. Bedrock credentials use the standard boto3
credential chain. Prefer short-lived SSO or workload credentials over static
keys.

## Batch container

The image installs from `uv.lock` and runs as an unprivileged user.

```bash
mkdir -p data reports
docker build -f docker/Dockerfile -t qualipilot:dev .
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD/data",dst=/data,readonly \
  --mount type=bind,src="$PWD/reports",dst=/reports \
  qualipilot:dev check /data/input.csv --output /reports/report.json
```

Build an optional provider or engine into the image with
`--build-arg EXTRAS=bedrock` (or another project extra). Supply cloud
credentials explicitly through your runtime's workload identity. The image
does not contain or mount credentials by default.

## Local Ollama compose stack

```bash
mkdir -p reports
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
docker compose -f docker/docker-compose.yml build qualipilot
docker compose -f docker/docker-compose.yml run --rm qualipilot
docker compose -f docker/docker-compose.yml down
```

The first run downloads the model and can take several minutes. Ollama is
bound to `127.0.0.1:11434`; it is not exposed on every host interface. Model
data stays in the `ollama-models` volume and generated output is written to
the repository's `reports/` directory.

## AWS Lambda

Prerequisites:

- Terraform 1.10 or newer
- AWS CLI and Docker authenticated to the intended AWS account
- permission to create S3, ECR, IAM, Lambda, SQS, and CloudWatch resources
- a remote Terraform state backend for shared or production environments

Configure remote state before the first apply. Copy
`deploy/terraform/backend.hcl.example` to the ignored `backend.hcl`, then
replace the bucket and region:

```hcl
bucket       = "existing-terraform-state-bucket"
key          = "qualipilot/production.tfstate"
region       = "us-east-1"
encrypt      = true
use_lockfile = true
```

The Terraform module declares the S3 backend without account-specific values.
Initialize it with `terraform init -backend-config=backend.hcl`. Do not commit
backend configuration or credentials.

### 1. Create the repository and supporting resources

An omitted `image_digest` intentionally skips Lambda on the bootstrap apply:

```bash
cd deploy/terraform
terraform init -backend-config=backend.hcl
terraform apply \
  -var project=qualipilot \
  -var region=us-east-1

ECR_URL="$(terraform output -raw ecr_repository_url)"
```

This avoids a first-apply dependency on an image that has not been pushed.

### 2. Build and push a unique image tag

```bash
REGISTRY="${ECR_URL%%/*}"
REPOSITORY="${ECR_URL##*/}"
TAG="v$(python -c 'import tomllib; print(tomllib.load(open("../../pyproject.toml", "rb"))["project"]["version"])')-$(git -C ../.. rev-parse --short=12 HEAD)"

test -z "$(git -C ../.. status --porcelain)"

aws ecr get-login-password --region us-east-1 |
  docker login --username AWS --password-stdin "$REGISTRY"

docker build --platform linux/amd64 \
  --build-arg "VCS_REF=$(git -C ../.. rev-parse HEAD)" \
  -f ../../docker/Dockerfile.lambda \
  -t "${ECR_URL}:${TAG}" ../..
docker push "${ECR_URL}:${TAG}"

DIGEST="$(aws ecr describe-images \
  --region us-east-1 \
  --repository-name "$REPOSITORY" \
  --image-ids "imageTag=$TAG" \
  --query 'imageDetails[0].imageDigest' \
  --output text)"
```

The repository rejects tag overwrites. Terraform deploys the returned digest
instead of a mutable tag.

### 3. Deploy Lambda

```bash
terraform apply \
  -var project=qualipilot \
  -var region=us-east-1 \
  -var "image_digest=$DIGEST" \
  -var 'alarm_action_arns=["arn:aws:sns:us-east-1:ACCOUNT:qualipilot-alerts"]' \
  -var 'qualipilot_config={engine="polars"}'
```

`alarm_action_arns` is required when Lambda is deployed; use a monitored SNS
topic or incident-management action.

`qualipilot_config` is a schema-validated HCL value. It exposes the
Lambda-supported
`auto` and `polars` engines, all non-linkage check settings, and the `none`
or `bedrock` LLM settings. Terraform rejects invalid bounds,
duplicate or empty column-list entries, and malformed range configuration
before deployment. Lambda freshness configuration uses UTC:

```hcl
qualipilot_config = {
  engine = "polars"
  checks = {
    min_rows                         = 1
    required_columns                 = ["order_id", "created_at"]
    freshness                        = true
    freshness_columns                = ["created_at"]
    freshness_timezone               = "UTC"
    freshness_max_age_hours          = 24
    freshness_future_tolerance_hours = 0
    column_ranges = {
      amount = {
        min = 0
        max = 100000
      }
    }
  }
}
```

Output paths, non-JSON formats, linkage, local LLM endpoints, and credentials
are intentionally absent from this deployment input.

If narrative reporting uses a foundation model directly, allow only its
exact ARN:

```hcl
bedrock_model_arns = [
  "arn:aws:bedrock:REGION::foundation-model/YOUR_MODEL"
]
```

For an inference profile, map its ARN to every source and destination
foundation-model ARN exposed by that profile:

```hcl
bedrock_inference_profiles = {
  "arn:aws:bedrock:REGION:ACCOUNT:inference-profile/YOUR_PROFILE" = [
    "arn:aws:bedrock:::foundation-model/YOUR_GLOBAL_MODEL",
    "arn:aws:bedrock:REGION_A::foundation-model/YOUR_MODEL",
    "arn:aws:bedrock:REGION_B::foundation-model/YOUR_MODEL",
  ]
}
```

Inference profiles also change where LLM metadata can be processed. A
geographic profile can move prompts and responses outside the source Region
while keeping them inside its named geography; data retained for abuse
detection can be stored in a destination Region. A global profile can route
requests across supported commercial AWS Regions worldwide. Review the AWS
[geographic](https://docs.aws.amazon.com/bedrock/latest/userguide/geographic-cross-region-inference.html)
and
[global](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html)
residency guidance and every destination Region before enabling a profile.
Use a directly invoked in-Region model when policy requires processing to
remain in the source Region.

When `qualipilot_config` enables Bedrock, its `llm.model` must be the
same exact model or inference-profile ARN supplied in one of these allowlists.

Bedrock reporting requires a Lambda timeout of at least 120 seconds. The
handler caps retries and reserves the final minute for report upload.

The model permissions are conditioned on the profile ARN, so they cannot be
used for direct invocation. Application inference-profile ARNs are also
accepted. Lambda accepts only `none` and `bedrock` providers. With both
variables empty, Lambda has no Bedrock permission.

### 4. Upload input

```bash
BUCKET="$(terraform output -raw data_bucket)"
aws s3 cp orders.parquet "s3://${BUCKET}/incoming/orders.parquet"
```

The S3 notification invokes Lambda. Versioned inputs produce immutable
report keys:

```text
s3://BUCKET/reports/incoming/orders.parquet.quality.VERSION_HASH.json
```

For unversioned or suspended buckets, Lambda binds the download to the
validated ETag with `If-Match` and enforces the byte limit while streaming.
An object changed between validation and download is rejected and retried.

Supported suffixes are `.csv`, `.jsonl`, `.ndjson`, `.parquet`,
and `.pq`. The default maximum object size is 256 MiB; set
`max_input_bytes` and `ephemeral_storage_mb` together after measuring actual
decompression and dataframe memory requirements. Lambda also rejects
inputs above 10,000 columns, nested values, Parquet data whose metadata
exceeds one third of configured function memory, and text inputs whose
conservative eight-times expansion estimate exceeds that memory budget.

Direct invocation is also supported for objects within the IAM-approved
bucket and prefix:

```json
{
  "s3_uri": "s3://BUCKET/incoming/orders.parquet",
  "config": {"engine": "polars"},
  "fail_on": "warn"
}
```

`fail_on` accepts `none`, `warn`, or `error`. Reports are uploaded first.
Crossing the threshold returns `quality_gate: "failed"` and emits the
`Qualipilot/QualityGateFailures` metric without classifying bad data as a
Lambda crash. Check-execution faults still fail the invocation and enter the
failure queue. Reports with a check-execution or LLM failure are retained
under `reports/failures/` but are not cached, so a replay can recover.
`output_key` names the terminal successful report; a failed attempt uses the
failure prefix. Direct invocation configuration overrides
`qualipilot_config`.

## Operations and security

- CloudWatch retains function logs for 30 days by default. The required
  `alarm_action_arns` route errors, throttles, duration, concurrency, delivery,
  quality, LLM, and failure-queue alarms. Duration alerts at 80% of the
  configured timeout and concurrency at 80% of the reserved limit.
- Deterministic asynchronous failures are sent directly to the SQS failure
  destination without a paid retry. Messages remain there for 14 days.
- Reserved concurrency defaults to five to bound account impact. Tune it
  with S3 arrival rate, object size, and account concurrency in mind.
- S3 public access is blocked, default encryption and versioning are enabled,
  and incomplete uploads and old noncurrent versions are expired.
- Lambda can read only the configured input prefix and can write only
  `reports/`. It cannot overwrite inputs.
- Keep `sample_size` and `include_top_values` disabled unless report
  consumers are authorized to see row-level or value-level data.
- Lambda rejects remote HTTP LLM providers and inline API credentials.
- Container base tags and dependencies still require routine rebuilds and
  vulnerability review. Treat a clean audit as point-in-time evidence.
- Back up and protect Terraform state; it contains resource identifiers and
  may contain sensitive values added by downstream customization.

### Failure recovery and rollback

The SQS queue is intentionally a human-reviewed failure destination, not an
automatic replay loop. For a failed message, inspect CloudWatch logs, copy
the event from the message's `requestPayload`, and invoke the function
synchronously. Delete the SQS message only after that invocation succeeds
and its report exists below `reports/`.

For a release, upload one representative object first and verify its report,
metrics, and logs before restoring normal traffic. To roll back, re-apply
Terraform with the previously approved ECR `image_digest`; never reuse or
overwrite an image tag.
