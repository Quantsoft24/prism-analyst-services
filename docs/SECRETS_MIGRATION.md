# Secrets & Config Migration — `.env`-over-SSH → AWS SSM Parameter Store

> **Goal:** stop hand-copying `.env` to EC2 on every deploy. Move secrets/config
> into **AWS SSM Parameter Store** (KMS-encrypted, versioned, audited), let the
> EC2 instance read them via an **IAM role**, and **render `.env` on the box at
> deploy time**. Docker Compose stays unchanged (still `env_file: .env`), and
> GitHub Actions never sees a secret.
>
> Status: **migration runbook — execute on AWS when ready.** Local development is
> unaffected (see [README → Local development](../README.md#local-development)).

---

## 0. Target architecture

```
            push: production
GitHub Actions ──SSH (EC2_HOST/USERNAME/SSH_KEY)──▶ EC2 instance
   (no AWS creds, no secrets)                         │
                                                      │  IAM instance role
                                                      ▼  (ssm:GetParametersByPath + kms:Decrypt)
                                          render-env-from-ssm.sh
                                                      │
                              aws ssm get-parameters-by-path /prism/prod/ --with-decryption
                                                      │
                                                      ▼
                       writes  ../prism-analyst-services/.env  (ephemeral, 0600, gitignored)
                                                      │
                                      docker compose up -d backend worker
```

**What changes vs today:** the only new moving parts are (a) parameters in SSM,
(b) an IAM role on the instance, (c) a 15-line render script, (d) ~4 lines added
to `deploy.yml`. Compose, the Dockerfile, and the app code are untouched.

**Why Parameter Store (not Secrets Manager):** SecureString parameters are
KMS-encrypted, versioned, and CloudTrail-audited, and the **standard tier is
free**. Secrets Manager (~$0.40/secret/mo) only earns its keep when you want
**built-in automatic rotation** — worth it later for the RDS passwords; see
[§9](#9-optional-upgrades). Start with Parameter Store.

---

## 1. Why move off the file-copy

The good part of today's setup: `.env` is gitignored (config is separated from
code — 12-factor). The problems are all in *how the file reaches the box*:

- **Secrets at rest + in transit** — they live on laptops and are `scp`'d to the
  instance. One leaked file = full compromise (DB, LLM keys, RDS).
- **Manual + every deploy** — error-prone; caused the `up -d --force-recreate`
  gotcha (compose doesn't re-read `.env` on a plain `restart`).
- **No versioning / rotation / audit** — nobody can answer "what's actually live
  and who changed it." SSM gives version history + CloudTrail on every read/write.
- **Config drift** — the box and the example file diverge silently.

---

## 2. Secret vs config inventory (what goes where)

Everything moves into SSM under `/prism/prod/<KEY>`. Mark true secrets as
**SecureString**; plain operational config can be **String** (cheaper, visible in
the console for debugging — but feel free to make everything SecureString for
uniformity; the render script decrypts both).

| Key | Type | Notes |
|---|---|---|
| `DATABASE_URL` | **SecureString** | Neon/RDS DSN with password |
| `GEMINI_API_KEY`, `GEMINI_API_KEY_1..4` | **SecureString** | LLM keys |
| `INVESTMENT_DB_PASSWORD` | **SecureString** | RDS (has `$` — store raw, no quotes) |
| `SEBI_DB_PASSWORD` | **SecureString** | |
| `OPENROUTER_API_KEY`, `TAVILY_API_KEY`, `PRISM_*_API_KEY`, `SUPABASE_JWT_SECRET` | **SecureString** | if/when set |
| `INVESTMENT_DB_HOST/PORT/NAME/USER`, `SEBI_DB_HOST/PORT/NAME/USER` | String | host/user aren't secrets, but SecureString is fine |
| `DB_SSL_MODE`, `INVESTMENT_DB_SSL_MODE`, `SEBI_DB_SSL_MODE` | String | |
| `BMC_URL`, `STOCK_CHAT_URL`, `PRISM_NEWS_URL`, `PRISM_FINANCIALS_URL`, `PRISM_FILINGS_URL` | String | service endpoints |
| `HOST`, `PORT`, `DEBUG`, `CORS_ORIGINS`, `API_PREFIX` | String | |
| `ADK_PROVIDER`, `MODEL_ROUTER_*`, `AGENT_*`, `AUTH_ENABLED`, `DEV_FIRM_ID` | String | |

> **Store raw values — no surrounding quotes.** Compose `env_file` passes each
> `KEY=VALUE` line literally to the container env and does **not** do shell
> interpolation, so `INVESTMENT_DB_PASSWORD` with a `$` is safe unquoted. (The
> single-quotes in `.env.example` are only to protect the value in a hand-edited
> dotenv; SSM stores the literal string.)

**Out of scope (not secrets, leave as-is):**
- Frontend `NEXT_PUBLIC_*` — baked as **build args** in `docker-compose.prod.yml`;
  public by design (they ship in the browser bundle), including the Supabase
  *publishable* anon key.
- `thequantsoft/.env` (landing) — apply the same pattern later under
  `/prism/prod/landing/` if it ever holds a secret; today it's minimal.
- `global-bundle.pem` (RDS CA bundle) — a public cert, not a secret. Keep it on
  the box (or store as a String param and write it out in the render script if
  you want it fully reproducible).

Region for all commands below: **`ap-south-1`** (Mumbai, same as RDS). Replace
`<ACCOUNT_ID>` with your AWS account id.

---

## 3. Migration — step by step

### Phase 0 — Prerequisites (one-time, on the EC2 box)

```bash
# AWS CLI v2 must be present on the instance (the render script uses it).
aws --version || sudo snap install aws-cli --classic   # or the official v2 installer
# jq for robust rendering:
jq --version || sudo apt-get update && sudo apt-get install -y jq
```

No AWS credentials are configured on the box — it will authenticate via the
**instance role** attached in Phase 2 (the SDK/CLI picks it up from instance
metadata automatically).

### Phase 1 — Create the KMS key (optional) and push parameters

You can use the AWS-managed key `alias/aws/ssm` for SecureString and skip the
custom key. A **customer-managed key** gives tighter access control + its own
rotation; recommended but optional.

```bash
# (Optional) customer-managed KMS key for PRISM secrets
aws kms create-key --description "PRISM prod SSM secrets" --region ap-south-1
aws kms create-alias --alias-name alias/prism-prod \
  --target-key-id <KEY_ID_FROM_ABOVE> --region ap-south-1
```

**Bootstrap from your existing `.env`** (run once, from a trusted machine that
has the current `.env` and AWS admin creds — NOT the EC2 box). This reads each
`KEY=VALUE` line and pushes it; secrets get `SecureString`, the rest `String`.

```bash
# scripts/bootstrap-ssm-from-env.sh  (one-time helper — delete after use)
#!/usr/bin/env bash
set -euo pipefail
REGION=ap-south-1
PREFIX=/prism/prod
KMS=alias/prism-prod          # or alias/aws/ssm

# Keys that must be SecureString (everything else → String).
SECRETS='DATABASE_URL GEMINI_API_KEY GEMINI_API_KEY_1 GEMINI_API_KEY_2 GEMINI_API_KEY_3 GEMINI_API_KEY_4 INVESTMENT_DB_PASSWORD SEBI_DB_PASSWORD OPENROUTER_API_KEY TAVILY_API_KEY SUPABASE_JWT_SECRET PRISM_FINANCIALS_API_KEY PRISM_NEWS_API_KEY'

while IFS= read -r line; do
  [[ "$line" =~ ^[[:space:]]*# || -z "${line// }" ]] && continue   # skip comments/blank
  key="${line%%=*}"; val="${line#*=}"
  val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"  # strip wrapping quotes
  [[ -z "$val" ]] && continue                                       # skip empty
  if grep -qw "$key" <<<"$SECRETS"; then type=SecureString; else type=String; fi
  echo "→ $PREFIX/$key ($type)"
  aws ssm put-parameter --name "$PREFIX/$key" --value "$val" --type "$type" \
    $( [[ "$type" == SecureString ]] && echo --key-id "$KMS" ) \
    --overwrite --region "$REGION" >/dev/null
done < .env
echo "Done. Verify: aws ssm get-parameters-by-path --path $PREFIX --recursive --region $REGION --query 'Parameters[].Name'"
```

```bash
bash scripts/bootstrap-ssm-from-env.sh
```

Verify (names only, no values):

```bash
aws ssm get-parameters-by-path --path /prism/prod --recursive \
  --region ap-south-1 --query 'Parameters[].Name' --output text
```

### Phase 2 — IAM instance role (least-privilege) and attach to EC2

Create a role the instance can assume, allowing **read-only** access to just the
`/prism/prod/*` params and decrypt with the key.

`prism-ssm-read-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadPrismParams",
      "Effect": "Allow",
      "Action": ["ssm:GetParametersByPath", "ssm:GetParameter", "ssm:GetParameters"],
      "Resource": "arn:aws:ssm:ap-south-1:<ACCOUNT_ID>:parameter/prism/prod/*"
    },
    {
      "Sid": "DecryptWithPrismKey",
      "Effect": "Allow",
      "Action": "kms:Decrypt",
      "Resource": "arn:aws:kms:ap-south-1:<ACCOUNT_ID>:alias/prism-prod"
    }
  ]
}
```
> If you used `alias/aws/ssm` instead of a custom key, scope `kms:Decrypt` to
> that key's ARN (or add a condition `kms:ViaService = ssm.ap-south-1.amazonaws.com`).

```bash
aws iam create-role --role-name prism-ec2-ssm \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name prism-ec2-ssm \
  --policy-name prism-ssm-read --policy-document file://prism-ssm-read-policy.json
aws iam create-instance-profile --instance-profile-name prism-ec2-ssm
aws iam add-role-to-instance-profile --instance-profile-name prism-ec2-ssm --role-name prism-ec2-ssm
# Attach to the running instance (no reboot needed):
aws ec2 associate-iam-instance-profile --region ap-south-1 \
  --instance-id <EC2_INSTANCE_ID> \
  --iam-instance-profile Name=prism-ec2-ssm
```

Confirm the box can read (on the EC2 instance):
```bash
aws ssm get-parameter --name /prism/prod/PRISM_NEWS_URL --region ap-south-1 --query 'Parameter.Value' --output text
```

### Phase 3 — Add the render script to the repo

Commit `scripts/render-env-from-ssm.sh` (runs on the box; writes `.env`):

```bash
#!/usr/bin/env bash
# Render ../prism-analyst-services/.env from SSM Parameter Store.
# Runs on EC2 using the instance role. Idempotent; overwrites .env atomically.
set -euo pipefail
REGION="${AWS_REGION:-ap-south-1}"
PREFIX="${SSM_PREFIX:-/prism/prod}"
OUT="${1:-$HOME/PRISM/prism-analyst-services/.env}"

tmp="$(mktemp)"
{
  echo "# GENERATED from SSM ${PREFIX} on $(date -u +%FT%TZ) — DO NOT EDIT BY HAND"
  aws ssm get-parameters-by-path \
      --path "$PREFIX" --recursive --with-decryption \
      --region "$REGION" --output json \
    | jq -r --arg p "$PREFIX/" '.Parameters[] | "\(.Name | ltrimstr($p))=\(.Value)"'
} > "$tmp"

# Sanity: must have rendered at least the DB URL.
grep -q '^DATABASE_URL=' "$tmp" || { echo "render failed: DATABASE_URL missing"; rm -f "$tmp"; exit 1; }

install -m 0600 "$tmp" "$OUT"   # 0600 = owner-only
rm -f "$tmp"
echo "Wrote $OUT ($(grep -c '=' "$OUT") vars)"
```

```bash
chmod +x scripts/render-env-from-ssm.sh
```

> **`.env` is now a generated artifact.** Keep it gitignored. Anyone (re)running
> the script regenerates it from the single source of truth (SSM).

### Phase 4 — Wire it into the deploy workflow

In `prism-analyst-services/.github/workflows/deploy.yml`, add a render step in
the SSH `script:` **before the build/up**, right after the `git reset` block:

```bash
            echo "🔐 Rendering .env from SSM Parameter Store..."
            bash ~/PRISM/prism-analyst-services/scripts/render-env-from-ssm.sh \
                 ~/PRISM/prism-analyst-services/.env
```

Then the existing `docker compose ... up -d backend` picks up the fresh `.env`.
**No GitHub secret changes** — CI still only needs `EC2_HOST` / `EC2_USERNAME` /
`EC2_SSH_KEY`. AWS auth happens on the box via the instance role.

> Compose re-reads `env_file` on `up` but **not** on a plain `restart`; the
> workflow already uses `up -d`, so this is correct. If you ever change a param
> and want it live without a full deploy: re-render, then
> `docker compose -f .../docker-compose.prod.yml up -d --force-recreate backend worker`.

### Phase 5 — Cut over, verify, then **rotate**

1. Run one deploy (push to `production`, or run the render script + `up -d`
   manually first to de-risk). Confirm `/health` is green and a news/chat call
   works.
2. **Rotate every secret that ever lived in a hand-copied `.env`** — they've been
   on laptops and over SSH, so treat them as exposed: regenerate Gemini keys,
   rotate the Neon/RDS/SEBI DB passwords, then `put-parameter --overwrite` the
   new values and re-deploy. (Good moment to also rotate the DSN that was
   hard-coded in `scripts/setup_company_aliases.py`.)
3. Delete local `.env` copies you no longer need and the one-time
   `bootstrap-ssm-from-env.sh`.

### Phase 6 — Decommission the ritual

- Stop hand-copying `.env`. The box regenerates it every deploy.
- Keep `.env.example` (it's the human-readable catalog of vars + the local-dev
  template). Add a one-liner pointing devs here for prod.

---

## 4. Rollback

The change is additive and reversible:
- Remove the render step from `deploy.yml` and copy a `.env` to the box again
  (old behavior). The instance role + SSM params are harmless if unused.
- Because `.env` is generated each deploy, a bad param is fixed by
  `put-parameter --overwrite` + re-render — no code change, no redeploy of the
  image required (`up -d --force-recreate backend worker`).

---

## 5. Day-2 operations

```bash
# Add / change a value
aws ssm put-parameter --name /prism/prod/GEMINI_API_KEY_2 --value "NEW" \
  --type SecureString --key-id alias/prism-prod --overwrite --region ap-south-1

# Inspect history (who/when — values hidden unless --with-decryption)
aws ssm get-parameter-history --name /prism/prod/DATABASE_URL --region ap-south-1 \
  --query 'Parameters[].[Version,LastModifiedDate,LastModifiedUser]' --output table

# Apply without a code deploy
ssh <box> 'bash ~/PRISM/prism-analyst-services/scripts/render-env-from-ssm.sh && \
  docker compose -f ~/PRISM/prism-analyst-platform/docker-compose.prod.yml up -d --force-recreate backend worker'
```

---

## 6. Local development

**Local dev does NOT use SSM or AWS at all.** Keep using a hand-written `.env`
from `.env.example` — see
[README → Local development](../README.md#local-development) and
[Environment variables](../README.md#environment-variables). This keeps the
local loop zero-dependency (no AWS account needed to run the backend).

*Optional parity:* if you maintain a separate `/prism/dev/` parameter tree, a dev
with `ssm:GetParametersByPath` on `/prism/dev/*` can pull non-prod values with
`SSM_PREFIX=/prism/dev scripts/render-env-from-ssm.sh ./.env`. Don't point local
at `/prism/prod`.

---

## 7. Staging / multi-env

Use the path prefix as the environment switch: `/prism/dev/`, `/prism/staging/`,
`/prism/prod/`. The same render script takes `SSM_PREFIX`; the same IAM policy
pattern scopes each instance to its own tree.

---

## 9. Optional upgrades (later)

- **Secrets Manager + rotation** for the DB passwords (RDS supports managed
  rotation Lambdas). Swap those few keys to Secrets Manager; the render script
  gains a couple of `aws secretsmanager get-secret-value` lines. Worth it once
  rotation cadence matters.
- **Fetch-at-boot inside the container** (entrypoint reads SSM) instead of
  rendering a file — removes `.env` from disk entirely. More moving parts (CLI
  in the image, metadata reachable from the container); the at-deploy render is
  simpler and already removes the SSH copy, so defer this.
- **Move off SSH entirely** — ECS Fargate task definitions reference Secrets
  Manager/SSM ARNs directly (`secrets:` block), so there's no box to render on.
  The natural endpoint if/when you outgrow the single EC2 + compose model.
```
