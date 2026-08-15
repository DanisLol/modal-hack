# modal-hack

EgoDiversity: a Modal-powered, reference-free diversity scoring pipeline for EgoVerse egocentric robot-learning data. See [PLAN.md](PLAN.md) for the full design.

## Quickstart

Set up the local dev environment and Modal auth:

```bash
pip install -r requirements-dev.txt
modal token new
```

Create the Modal Secret holding EgoVerse AWS/DB credentials (run once, by whoever has real credentials; required before `healthcheck` reports secrets as present):

```bash
modal secret create egoverse-creds \
  AWS_ACCESS_KEY_ID=... \
  AWS_SECRET_ACCESS_KEY=... \
  AWS_REGION=... \
  DB_URL=...
```

Sanity-check the App/Volume/Secret skeleton:

```bash
modal run -m egodiversity.app
```

Deploy the app:

```bash
modal deploy -m egodiversity.app
```