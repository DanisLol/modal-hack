#!/usr/bin/env bash
# Create/update the `egoverse-creds` Modal Secret from EgoVerse's own
# credential file (~/.egoverse_env), produced by their
# `egomimic/utils/aws/setup_secret.sh` script.
#
# One-time prerequisite (run inside a clone of GaTech-RL2/EgoVerse):
#   aws configure                              # use the demo key pair from the EgoVerse README
#   bash egomimic/utils/aws/setup_secret.sh    # writes ~/.egoverse_env
#
# Then, from this project:
#   ./scripts/push_egoverse_secret.sh
#
# This never checks credentials into the repo: it reads the local
# ~/.egoverse_env file (0600, git-ignored by convention) and uploads its
# values straight to Modal's secret store.
set -euo pipefail

ENV_FILE="${1:-$HOME/.egoverse_env}"
SECRET_NAME="${SECRET_NAME:-egoverse-creds}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: $ENV_FILE not found." >&2
  echo "Run EgoVerse's setup_secret.sh first (see script header for the two commands)." >&2
  exit 1
fi

if ! command -v modal >/dev/null 2>&1; then
  echo "error: the 'modal' CLI is not installed. Run 'pip install modal' first." >&2
  exit 1
fi

TMP_JSON="$(mktemp)"
trap 'rm -f "$TMP_JSON"' EXIT

# $ENV_FILE holds bash `printf %q`-escaped KEY=value lines. Source it in a
# throwaway subshell and re-emit only the keys EgoDiversity cares about, so
# we hand Modal already-unescaped values regardless of %q's quoting style.
python3 - "$ENV_FILE" "$TMP_JSON" <<'PY'
import json
import subprocess
import sys

env_file, out_path = sys.argv[1], sys.argv[2]

keys = [
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_SESSION_TOKEN",
    "AWS_ENDPOINT_URL_S3",
    "AWS_DEFAULT_REGION",
    "BUCKET",
    "SECRETS_ARN",
]

script = f'set -a; source "{env_file}" >/dev/null 2>&1; ' + " ".join(
    f'echo "{k}=${{{k}:-}}"' for k in keys
)
proc = subprocess.run(
    ["bash", "-c", script], capture_output=True, text=True, check=True
)

values = {}
for line in proc.stdout.splitlines():
    if "=" not in line:
        continue
    key, value = line.split("=", 1)
    if value:
        values[key] = value

required = {"R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL_S3", "BUCKET"}
missing = required - values.keys()
if missing:
    print(f"error: missing required keys in {env_file}: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)

with open(out_path, "w") as f:
    json.dump(values, f)

print(f"Resolved {len(values)} credential keys from {env_file}: {', '.join(sorted(values))}")
PY

modal secret create "$SECRET_NAME" --from-json "$TMP_JSON" --force
echo "Created/updated Modal secret '$SECRET_NAME'."
echo "Verify with: modal run -m egodiversity.app"
