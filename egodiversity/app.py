"""Central Modal orchestration module for EgoDiversity.

Defines the shared `modal.App`, container `Image`, cache `Volume`, and
credentials `Secret` that every later pipeline stage (ingestion, feature
extraction, metrics, API) attaches to. Other modules should import
`app`, `image`, `volume`, and the path constants from here rather than
redefining them, so the whole project shares one App and one cache.
"""

import modal

APP_NAME = "egodiversity"
VOLUME_NAME = "egodiversity-cache"
SECRET_NAME = "egoverse-creds"

# Env vars expected in the `egoverse-creds` Modal Secret, matching exactly
# what EgoVerse's own `egomimic/utils/aws/setup_secret.sh` writes to
# `~/.egoverse_env` (see CONTRIBUTING_DATA.md / README "AWS Configure").
# Use `scripts/push_egoverse_secret.sh` to create this secret from that file.
#
# R2_* / *_ENDPOINT_URL / BUCKET are always present (R2 object storage
# access, works for both the internal and public-fallback credential tiers).
REQUIRED_SECRET_KEYS = [
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL_S3",
    "AWS_DEFAULT_REGION",
    "BUCKET",
]
# SECRETS_ARN is only written when the RL2-internal (or public read-only)
# Postgres secret was reachable; R2_SESSION_TOKEN only appears for
# session-scoped (STS) R2 credentials. Both are optional.
OPTIONAL_SECRET_KEYS = [
    "SECRETS_ARN",
    "R2_SESSION_TOKEN",
]
EXPECTED_SECRET_KEYS = REQUIRED_SECRET_KEYS + OPTIONAL_SECRET_KEYS

# Volume-backed cache layout, shared by all pipeline stages.
CACHE_ROOT = "/cache"
EPISODES_DIR = f"{CACHE_ROOT}/episodes"
EMBEDDINGS_DIR = f"{CACHE_ROOT}/embeddings"
RESULTS_DIR = f"{CACHE_ROOT}/results"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "boto3",
    "sqlalchemy",
    "psycopg2-binary",
    "numpy",
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

secret = modal.Secret.from_name(SECRET_NAME)

app = modal.App(APP_NAME, image=image, secrets=[secret])


@app.function(volumes={CACHE_ROOT: volume})
def healthcheck() -> dict:
    """Verify Modal App/Volume/Secret wiring.

    Creates the cache subdirectories on the mounted Volume, commits the
    Volume, and reports which expected credential env vars are present
    (without leaking their values), flagging whether any required key is
    missing. Used as a smoke test that the App, Volume, and Secret are all
    correctly configured before any real pipeline stage is built on top of
    them.
    """
    import os

    for directory in (EPISODES_DIR, EMBEDDINGS_DIR, RESULTS_DIR):
        os.makedirs(directory, exist_ok=True)
    volume.commit()

    secrets_present = {key: key in os.environ for key in EXPECTED_SECRET_KEYS}
    missing_required = [key for key in REQUIRED_SECRET_KEYS if not secrets_present[key]]

    return {
        "app": APP_NAME,
        "volume_root": CACHE_ROOT,
        "cache_dirs": [EPISODES_DIR, EMBEDDINGS_DIR, RESULTS_DIR],
        "secrets_present": secrets_present,
        "missing_required_secrets": missing_required,
        "ok": not missing_required,
    }


@app.local_entrypoint()
def main() -> None:
    """Local CLI entrypoint: `modal run -m egodiversity.app` runs the healthcheck."""
    result = healthcheck.remote()
    print(result)
