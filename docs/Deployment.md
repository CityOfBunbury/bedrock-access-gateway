# Deployment with Docker Compose

This guide covers running the gateway on a single host (for example an EC2 instance) with the
`docker-compose.yml` in the repository root.

The compose file builds `src/Dockerfile_ecs`, which runs the app as a non-root user and has a
container health check on `/health`. The app listens on port 8080 inside the container (the
`PORT` value set in the Dockerfile) and compose publishes it on port 8000 of the host.

## Configuration

Compose reads these values from the shell or from a `.env` file next to `docker-compose.yml`
(`.env` is ignored by git):

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | The API key clients must send as a bearer token. Passed to the app as `API_KEY`. |
| `AWS_REGION` | The Region the gateway calls Bedrock in. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_PROFILE` | Optional. Only needed when the host has no instance role. See [Switch to an instance role](#switch-from-access-keys-to-an-instance-role). |

The compose file also sets `ENABLE_PROMPT_CACHING=true` and `ENABLE_IMAGE_URL_FETCH=false`
(images must be sent inline as base64 data URLs). Other settings are listed in `src/api/setting.py`.

Restrict the published port to the clients that need it, for example with a security group
that only allows port 8000 from the client host.

## Deploy

```bash
git pull
# Keep the image that is running now so you can roll back to it.
docker tag bedrock-access-gateway:latest bedrock-access-gateway:previous
docker compose build
docker compose up -d
```

Then check that the container is healthy and answers:

```bash
docker compose ps                      # STATUS should show "(healthy)" after a few seconds
curl -s http://localhost:8000/health   # {"status":"OK"}
curl -s http://localhost:8000/api/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

The first time you deploy from a version whose compose file had no `image:` line, the running
image has a name chosen by compose. Find it with `docker compose images` and tag that image as
`bedrock-access-gateway:previous` before you build.

## Roll back

Put the previous image back under the name compose runs, and restart without building:

```bash
docker tag bedrock-access-gateway:previous bedrock-access-gateway:latest
docker compose up -d --no-build
```

You can also check out the previous commit and run `docker compose up -d --build`. That rebuilds
the image, so operating system packages and unpinned dependencies may differ from the image
that ran before.

## Switch from access keys to an instance role

The gateway creates its AWS clients without explicit credentials, so it uses the instance role
when no keys are passed in. No code change is needed.

1. Attach an IAM role to the instance with the Bedrock permissions below.
2. If containers use the default bridge network, set the instance metadata hop limit to 2.
   The container adds a network hop, so with a limit of 1 the IMDSv2 token response never
   reaches it and the SDK cannot get the role's credentials:

   ```bash
   aws ec2 modify-instance-metadata-options --instance-id <instance-id> \
     --http-put-response-hop-limit 2 --http-endpoint enabled
   ```

3. In `docker-compose.yml`, remove the `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` lines and the `~/.aws` volume. Keep
   `AWS_REGION`. Remove the keys from `.env` and the shell profile too, so they cannot be
   picked up again.
4. Run `docker compose up -d` and check `/api/v1/models` as above. Then deactivate and delete
   the old access keys.

### Bedrock permissions

The role needs to list models and inference profiles, and to call models through Converse
and ConverseStream:

| Action | Used for |
|--------|----------|
| `bedrock:ListFoundationModels` | Building the model list |
| `bedrock:ListInferenceProfiles` | Listing cross-Region and application inference profiles |
| `bedrock:InvokeModel` | Converse (non-streaming chat) and embeddings |
| `bedrock:InvokeModelWithResponseStream` | ConverseStream (streaming chat) |

The list actions need `"Resource": "*"`. Scope the invoke actions to the models you use. When
you call a model through an inference profile (IDs that start with `global.`, `us.`, `eu.`,
`apac.` and so on), allow the inference profile ARN and the foundation model ARNs it routes to
in every destination Region; global profiles also need the global foundation model ARN. See
"Prerequisites for inference profiles" in the Amazon Bedrock user guide for the exact resource
formats.

Example, with placeholders:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListModels",
      "Effect": "Allow",
      "Action": ["bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"],
      "Resource": "*"
    },
    {
      "Sid": "InvokeModels",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:<region>:<account-id>:inference-profile/<profile-id>",
        "arn:aws:bedrock:*::foundation-model/<model-id>",
        "arn:aws:bedrock:::foundation-model/<model-id>"
      ]
    }
  ]
}
```

If the API key is read from Secrets Manager (`API_KEY_SECRET_ARN`) or Parameter Store
(`API_KEY_PARAM_NAME`) instead of `API_KEY`, the role also needs
`secretsmanager:GetSecretValue` or `ssm:GetParameter` on that secret or parameter.
