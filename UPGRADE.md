# Competition Upgrade Tracker

This file tracks the status of every upgrade phase for the agri-yield competition submission.
All work happens on branch `improve/competition-upgrade` and is merged to `main` via PR.

---

## Phase 1 — Baseline & Stability ✅ In Progress

| Task | Status | File(s) |
|---|---|---|
| Create upgrade branch | ✅ Done | `improve/competition-upgrade` |
| DRY quality gate script | ✅ Done | `scripts/quality_gate.py` |
| CI updated to use shared script | ✅ Done | `.github/workflows/ci.yml` |
| Deploy updated to use shared script | ✅ Done | `.github/workflows/deploy.yml` |
| CI now runs on upgrade branch too | ✅ Done | `.github/workflows/ci.yml` |
| Coverage threshold raised 60 → 70 | ✅ Done | `.github/workflows/ci.yml` |
| `pytest-asyncio` added + auto mode | ✅ Done | `pyproject.toml` |
| `shap` added to dependencies | ✅ Done | `pyproject.toml` |
| Unimplemented deps commented out | ✅ Done | `pyproject.toml` |
| Dead-code stubs: features/, orchestration/, infra/ | ✅ Done | `*/README.md` |
| Helm chart documented | ✅ Done | `helm/README.md` |
| Screenshot directory created | ✅ Done | `docs/screenshots/` |
| Phase tracking issue created | ✅ Done | GitHub Issue #8 |

**Remaining (do locally):**
- [ ] Take baseline dashboard screenshot → save to `docs/screenshots/baseline-dashboard.png`
- [ ] Capture baseline `/health` JSON → save to `docs/screenshots/baseline-health.json`
- [ ] Run full test suite locally and record pass rate
- [ ] Run `uv lock` after `pyproject.toml` changes and commit updated `uv.lock`

---

## Phase 2 — Real Confidence Intervals 🕒 Next

| Task | Status | File(s) |
|---|---|---|
| Train lower/mean/upper quantile models | ⏳ Pending | `training/train_and_export.py` |
| Save 3-model bundle (`model_bundle.pkl`) | ⏳ Pending | `training/train_and_export.py` |
| Update `serving/model.py` to load bundle | ⏳ Pending | `serving/model.py` |
| Quality gate validates all 3 models | ⏳ Pending | `scripts/quality_gate.py` |
| Test: lower ≤ mean ≤ upper always | ⏳ Pending | `tests/test_model_bundle.py` |

---

## Phase 3 — SHAP Explainability 🕒 Planned

| Task | Status | File(s) |
|---|---|---|
| Global feature importance at training time | ⏳ Pending | `training/train_and_export.py` |
| `GET /model/info` endpoint | ⏳ Pending | `serving/app.py` |
| `POST /predict/explain` endpoint | ⏳ Pending | `serving/app.py` |
| SHAP waterfall chart in dashboard UI | ⏳ Pending | `serving/templates/` |
| Test: explain response shape | ⏳ Pending | `tests/test_api.py` |

---

## Phase 4 — Redis Drift Persistence + Retraining Trigger 🕒 Planned

| Task | Status | File(s) |
|---|---|---|
| Move `_live_buffer` to Redis lists | ⏳ Pending | `monitoring/psi_detector.py` |
| Cache reference distributions at startup | ⏳ Pending | `monitoring/psi_detector.py` |
| Add `workflow_dispatch` retraining trigger | ⏳ Pending | `.github/workflows/retrain.yml` |
| Test: drift buffer survives simulated restart | ⏳ Pending | `tests/test_drift.py` |

---

## Phase 5 — API Hardening + Deep Tests 🕒 Planned

| Task | Status | File(s) |
|---|---|---|
| Stricter Pydantic response models | ⏳ Pending | `serving/app.py` |
| Weather API timeout test | ⏳ Pending | `tests/test_api.py` |
| Redis unavailable failover test | ⏳ Pending | `tests/test_api.py` |
| Missing model file test | ⏳ Pending | `tests/test_api.py` |
| Bulk endpoint partial failure test | ⏳ Pending | `tests/test_api.py` |
| Coverage gate raised to 80% | ⏳ Pending | `.github/workflows/ci.yml` |

---

## Phase 6 — README + Demo Polish 🕒 Planned

| Task | Status | File(s) |
|---|---|---|
| Architecture diagram image | ⏳ Pending | `docs/architecture.png` |
| Real dashboard screenshot | ⏳ Pending | `docs/screenshots/dashboard.png` |
| Demo GIF | ⏳ Pending | `docs/screenshots/demo.gif` |
| Model card summary in README | ⏳ Pending | `README.md` |
| Metrics table (RMSE, latency, fields) | ⏳ Pending | `README.md` |
| Upgrade Render to paid tier for demo day | ⏳ Pending | `render.yaml` |

---

## Phase 7 — Advanced Features 🕒 Optional

| Task | Status | Notes |
|---|---|---|
| Great Expectations data validation | ⏳ Optional | High credibility signal |
| MLflow registry-first workflow | ⏳ Optional | Replace pickle-first fallback |
| Prefect scheduled retraining | ⏳ Optional | Full orchestration loop |
| Feast feature store | ⏳ Optional | Online/offline parity |
| Kafka stream ingestion | ⏳ Optional | Real-time event pipeline |
