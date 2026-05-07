"""
serving/app.py
FastAPI prediction endpoint for agri-yield.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ingestion.openmeteo_live import get_live_features
from monitoring.prometheus_metrics import (
    PREDICTION_CI_WIDTH,
    PREDICTION_LATENCY,
    PREDICTIONS_TOTAL,
    PREDICTION_YIELD_KG_HA,
    STALE_FEATURE_REQUESTS,
)
from monitoring.psi_detector import evaluate_drift
from serving.metrics import metrics_router
import serving.model as model_module
from serving.model import load_model
from serving.schemas import PredictRequest

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FIELDS_PATH = str(_REPO_ROOT / "data" / "seed" / "uk_fields.csv")
FIELDS_CSV_PATH = os.getenv("FIELDS_CSV_PATH", _DEFAULT_FIELDS_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and field metadata at startup."""
    global _FIELDS_DF
    try:
        log.info("Loading fields from: %s", FIELDS_CSV_PATH)
        _FIELDS_DF = pd.read_csv(FIELDS_CSV_PATH).set_index("field_id")
        log.info("Loaded %d fields", len(_FIELDS_DF))
    except FileNotFoundError:
        log.warning("%s not found — /predict will 503.", FIELDS_CSV_PATH)
        _FIELDS_DF = pd.DataFrame()

    load_model()
    yield


app = FastAPI(title="Agri Yield API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "https://hulashc.github.io",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metrics_router)

_FIELDS_DF: pd.DataFrame = pd.DataFrame()


def get_field_meta(field_id: str) -> dict:
    if _FIELDS_DF.empty or field_id not in _FIELDS_DF.index:
        raise HTTPException(status_code=404, detail=f"Unknown field_id: {field_id}")
    return _FIELDS_DF.loc[field_id].to_dict()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def map_ui():
    """Interactive Leaflet map showing all UK fields with on-click predictions."""
    if _FIELDS_DF.empty:
        return HTMLResponse("<h2>Fields not loaded</h2>", status_code=503)

    fields_json = []
    for fid, row in _FIELDS_DF.iterrows():
        fields_json.append({
            "id": str(fid),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "crop": str(row.get("crop_type", "unknown")),
            "region": str(row.get("region", "")),
            "area": round(float(row.get("area_ha", 0)), 1),
        })

    import json
    fields_js = json.dumps(fields_json)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agri Yield — UK Field Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', system-ui, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      height: 100dvh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 20px;
      background: #161b27;
      border-bottom: 1px solid #1e2a3a;
      z-index: 1000;
      flex-shrink: 0;
    }}
    .logo {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .logo svg {{ color: #4ade80; }}
    .logo-text {{ font-size: 1rem; font-weight: 600; letter-spacing: -0.01em; color: #f1f5f9; }}
    .logo-sub {{ font-size: 0.75rem; color: #64748b; margin-top: 1px; }}
    .status-pill {{
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      color: #94a3b8;
      background: #1e2a3a;
      padding: 5px 12px;
      border-radius: 999px;
      border: 1px solid #1e2a3a;
    }}
    .dot {{
      width: 7px; height: 7px;
      border-radius: 50%;
      background: #4ade80;
      box-shadow: 0 0 6px #4ade80;
    }}
    #map {{ flex: 1; }}
    /* Sidebar panel */
    #panel {{
      position: absolute;
      top: 72px;
      right: 16px;
      width: 300px;
      background: #161b27;
      border: 1px solid #1e2a3a;
      border-radius: 12px;
      z-index: 999;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      overflow: hidden;
      transition: opacity 0.2s ease;
      display: none;
    }}
    #panel.visible {{ display: block; }}
    .panel-header {{
      padding: 14px 16px 10px;
      border-bottom: 1px solid #1e2a3a;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }}
    .panel-field-id {{
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #4ade80;
    }}
    .panel-crop {{
      font-size: 1rem;
      font-weight: 600;
      color: #f1f5f9;
      margin-top: 2px;
      text-transform: capitalize;
    }}
    .panel-close {{
      background: none;
      border: none;
      color: #64748b;
      cursor: pointer;
      font-size: 1.1rem;
      padding: 2px 6px;
      border-radius: 6px;
      line-height: 1;
    }}
    .panel-close:hover {{ color: #f1f5f9; background: #1e2a3a; }}
    .panel-meta {{
      padding: 12px 16px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      border-bottom: 1px solid #1e2a3a;
    }}
    .meta-item {{ display: flex; flex-direction: column; gap: 2px; }}
    .meta-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.07em; color: #64748b; }}
    .meta-value {{ font-size: 0.875rem; color: #cbd5e1; font-weight: 500; }}
    .panel-result {{
      padding: 14px 16px;
    }}
    .yield-value {{
      font-size: 2rem;
      font-weight: 700;
      color: #f1f5f9;
      letter-spacing: -0.03em;
      line-height: 1;
    }}
    .yield-unit {{
      font-size: 0.75rem;
      color: #64748b;
      margin-top: 3px;
    }}
    .yield-range {{
      font-size: 0.75rem;
      color: #94a3b8;
      margin-top: 6px;
    }}
    .yield-bar-wrap {{
      margin-top: 10px;
      background: #1e2a3a;
      border-radius: 999px;
      height: 6px;
      overflow: hidden;
    }}
    .yield-bar {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #22c55e, #4ade80);
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .drift-badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      margin-top: 10px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .drift-none {{ background: #14532d22; color: #4ade80; border: 1px solid #14532d; }}
    .drift-low {{ background: #78350f22; color: #fbbf24; border: 1px solid #78350f; }}
    .drift-high {{ background: #7f1d1d22; color: #f87171; border: 1px solid #7f1d1d; }}
    .loading-spinner {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: #64748b;
      font-size: 0.8rem;
      padding: 14px 16px;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .spinner {{ width: 14px; height: 14px; border: 2px solid #1e2a3a; border-top-color: #4ade80; border-radius: 50%; animation: spin 0.7s linear infinite; }}
    /* Legend */
    #legend {{
      position: absolute;
      bottom: 30px;
      left: 16px;
      background: #161b27;
      border: 1px solid #1e2a3a;
      border-radius: 10px;
      padding: 10px 14px;
      z-index: 999;
      font-size: 0.7rem;
      color: #94a3b8;
    }}
    #legend .leg-title {{ font-weight: 600; color: #cbd5e1; margin-bottom: 7px; letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.65rem; }}
    .leg-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
    .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
    /* Leaflet dark overrides */
    .leaflet-container {{ background: #0f1117; }}
    .leaflet-tile {{ filter: brightness(0.6) saturate(0.7); }}
    .leaflet-popup-content-wrapper {{
      background: #161b27;
      border: 1px solid #1e2a3a;
      border-radius: 8px;
      color: #e2e8f0;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }}
    .leaflet-popup-tip {{ background: #161b27; }}
    .leaflet-popup-content {{ margin: 8px 12px; font-size: 0.8rem; }}
    .leaflet-control-zoom a {{ background: #161b27 !important; color: #94a3b8 !important; border-color: #1e2a3a !important; }}
    .leaflet-control-zoom a:hover {{ background: #1e2a3a !important; color: #f1f5f9 !important; }}
    .leaflet-control-attribution {{ background: #161b27aa !important; color: #475569 !important; font-size: 0.6rem; }}
    .leaflet-control-attribution a {{ color: #4ade80 !important; }}
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-label="Agri Yield">
        <rect width="28" height="28" rx="7" fill="#14532d"/>
        <path d="M14 6 C14 6 8 10 8 16 C8 19.3 10.7 22 14 22 C17.3 22 20 19.3 20 16 C20 10 14 6 14 6Z" fill="#4ade80" opacity="0.9"/>
        <path d="M14 22 L14 13" stroke="#14532d" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M14 17 L11 14" stroke="#14532d" stroke-width="1.2" stroke-linecap="round"/>
        <path d="M14 15 L17 12" stroke="#14532d" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
      <div>
        <div class="logo-text">Agri Yield</div>
        <div class="logo-sub">UK Field Predictions</div>
      </div>
    </div>
    <div class="status-pill">
      <div class="dot"></div>
      <span>Model live</span>
    </div>
  </header>

  <div id="map"></div>

  <!-- Side panel -->
  <div id="panel">
    <div class="panel-header">
      <div>
        <div class="panel-field-id" id="p-id">—</div>
        <div class="panel-crop" id="p-crop">—</div>
      </div>
      <button class="panel-close" onclick="closePanel()">✕</button>
    </div>
    <div class="panel-meta">
      <div class="meta-item"><span class="meta-label">Region</span><span class="meta-value" id="p-region">—</span></div>
      <div class="meta-item"><span class="meta-label">Area</span><span class="meta-value" id="p-area">—</span></div>
      <div class="meta-item"><span class="meta-label">Lat</span><span class="meta-value" id="p-lat">—</span></div>
      <div class="meta-item"><span class="meta-label">Lon</span><span class="meta-value" id="p-lon">—</span></div>
    </div>
    <div id="panel-result">
      <div class="loading-spinner"><div class="spinner"></div> Fetching prediction…</div>
    </div>
  </div>

  <!-- Legend -->
  <div id="legend">
    <div class="leg-title">Crop Type</div>
    <div class="leg-row"><div class="leg-dot" style="background:#4ade80"></div>Wheat</div>
    <div class="leg-row"><div class="leg-dot" style="background:#facc15"></div>Barley</div>
    <div class="leg-row"><div class="leg-dot" style="background:#fb923c"></div>Oilseed Rape</div>
    <div class="leg-row"><div class="leg-dot" style="background:#a78bfa"></div>Potato</div>
    <div class="leg-row"><div class="leg-dot" style="background:#38bdf8"></div>Sugar Beet</div>
    <div class="leg-row"><div class="leg-dot" style="background:#94a3b8"></div>Other</div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const FIELDS = {fields_js};

    const CROP_COLORS = {{
      wheat:        '#4ade80',
      barley:       '#facc15',
      oilseed_rape: '#fb923c',
      'oilseed rape': '#fb923c',
      potato:       '#a78bfa',
      sugar_beet:   '#38bdf8',
      'sugar beet':  '#38bdf8',
    }};

    function cropColor(crop) {{
      return CROP_COLORS[crop.toLowerCase()] || '#94a3b8';
    }}

    function makeIcon(crop) {{
      const c = cropColor(crop);
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 22 22">
        <circle cx="11" cy="11" r="8" fill="${{c}}" fill-opacity="0.25" stroke="${{c}}" stroke-width="2"/>
        <circle cx="11" cy="11" r="4" fill="${{c}}"/>
      </svg>`;
      return L.divIcon({{
        html: svg,
        className: '',
        iconSize: [22, 22],
        iconAnchor: [11, 11],
        popupAnchor: [0, -14],
      }});
    }}

    const map = L.map('map', {{
      center: [52.5, -1.5],
      zoom: 6,
      zoomControl: true,
    }});

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19,
    }}).addTo(map);

    let activeMarker = null;

    FIELDS.forEach(f => {{
      const marker = L.marker([f.lat, f.lon], {{ icon: makeIcon(f.crop) }}).addTo(map);
      marker.on('click', () => openPanel(f, marker));
    }});

    function openPanel(f, marker) {{
      if (activeMarker) activeMarker.setZIndexOffset(0);
      activeMarker = marker;
      marker.setZIndexOffset(1000);

      document.getElementById('p-id').textContent = f.id;
      document.getElementById('p-crop').textContent = f.crop.replace(/_/g, ' ');
      document.getElementById('p-region').textContent = f.region || '—';
      document.getElementById('p-area').textContent = f.area + ' ha';
      document.getElementById('p-lat').textContent = f.lat.toFixed(4);
      document.getElementById('p-lon').textContent = f.lon.toFixed(4);

      const panel = document.getElementById('panel');
      panel.classList.add('visible');
      document.getElementById('panel-result').innerHTML =
        '<div class="loading-spinner"><div class="spinner"></div> Fetching prediction…</div>';

      fetch('/predict', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ field_id: f.id }}),
      }})
        .then(r => r.json())
        .then(d => renderResult(d))
        .catch(() => {{
          document.getElementById('panel-result').innerHTML =
            '<div class="loading-spinner" style="color:#f87171">⚠ Prediction failed</div>';
        }});
    }}

    function renderResult(d) {{
      const mean = d.predicted_yield_kg_ha;
      const lo   = d.lower_bound;
      const hi   = d.upper_bound;
      const MAX_YIELD = 12000;
      const pct = Math.min(100, Math.round((mean / MAX_YIELD) * 100));

      const driftClass = d.drift_level === 'high' ? 'drift-high'
                       : d.drift_level === 'low'  ? 'drift-low'
                       : 'drift-none';
      const driftIcon  = d.drift_level === 'high' ? '⚠' : d.drift_level === 'low' ? '△' : '✓';
      const driftLabel = d.drift_warning ? `Drift: ${{d.drift_level}}` : 'No drift';

      document.getElementById('panel-result').innerHTML = `
        <div class="panel-result">
          <div class="yield-value">${{mean ? mean.toLocaleString(undefined, {{maximumFractionDigits: 0}}) : '—'}}</div>
          <div class="yield-unit">kg / ha predicted yield</div>
          <div class="yield-range">80% CI: ${{lo ? lo.toLocaleString(undefined, {{maximumFractionDigits:0}}) : '—'}} – ${{hi ? hi.toLocaleString(undefined, {{maximumFractionDigits:0}}) : '—'}} kg/ha</div>
          <div class="yield-bar-wrap"><div class="yield-bar" style="width:${{pct}}%"></div></div>
          <div class="drift-badge ${{driftClass}}">${{driftIcon}} ${{driftLabel}}</div>
        </div>`;
    }}

    function closePanel() {{
      document.getElementById('panel').classList.remove('visible');
    }}
  </script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model_module.is_loaded(),
        "model_version": model_module.model_version(),
        "fields_loaded": not _FIELDS_DF.empty,
    }


@app.post("/predict")
async def predict(request: PredictRequest) -> dict:
    if not model_module.is_loaded():
        raise HTTPException(status_code=503, detail="No model available.")

    start = time.time()
    field_meta = get_field_meta(request.field_id)
    live = get_live_features(
        request.field_id,
        float(field_meta["lat"]),
        float(field_meta["lon"]),
    )

    if live.get("stale_features"):
        STALE_FEATURE_REQUESTS.labels(field_id=request.field_id).inc()

    features = {**field_meta, **live}
    drift_result = evaluate_drift(request.field_id, features)
    feat_df = pd.DataFrame([features])
    preds, lower, upper = model_module.predict(feat_df)
    prediction = {
        "mean": float(preds[0]),
        "lower": float(lower[0]),
        "upper": float(upper[0]),
    }

    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)
    PREDICTIONS_TOTAL.labels(
        crop_type=str(field_meta["crop_type"]),
        region=str(field_meta["region"]),
    ).inc()
    PREDICTION_YIELD_KG_HA.labels(crop_type=str(field_meta["crop_type"])).observe(
        prediction["mean"]
    )
    PREDICTION_CI_WIDTH.observe(prediction["upper"] - prediction["lower"])

    return {
        "field_id": request.field_id,
        "predicted_yield_kg_ha": prediction["mean"],
        "lower_bound": prediction["lower"],
        "upper_bound": prediction["upper"],
        "confidence_level": 0.80,
        "drift_warning": drift_result["drift_warning"],
        "drift_level": drift_result["drift_level"],
        "psi_score": drift_result["max_psi"],
        "stale_features": live.get("stale_features", False),
        "model_version": model_module.model_version(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
