"""
serving/app.py
FastAPI prediction endpoint for agri-yield.
"""

import asyncio
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

# Limit concurrent Open-Meteo calls — free Render has 1 CPU and
# firing 100+ simultaneous HTTP requests causes most to timeout.
_PREDICT_SEMAPHORE = asyncio.Semaphore(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _FIELDS_DF
    try:
        log.info("Loading fields from: %s", FIELDS_CSV_PATH)
        _FIELDS_DF = pd.read_csv(FIELDS_CSV_PATH).set_index("field_id")
        log.info("Loaded %d fields", len(_FIELDS_DF))
    except FileNotFoundError:
        log.warning("%s not found.", FIELDS_CSV_PATH)
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


async def _predict_one(field_id: str, row: pd.Series) -> dict:
    """Run a single field prediction — used by bulk /fields endpoint."""
    async with _PREDICT_SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            live = await loop.run_in_executor(
                None,
                get_live_features,
                field_id,
                float(row["lat"]),
                float(row["lon"]),
            )
            features = {**row.to_dict(), **live}
            drift_result = evaluate_drift(field_id, features)
            feat_df = pd.DataFrame([features])
            preds, lower, upper = model_module.predict(feat_df)
            return {
                "field_id": field_id,
                "name": str(row.get("name", field_id)),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "crop_type": str(row.get("crop_type", "")),
                "region": str(row.get("region", "")),
                "area_ha": round(float(row.get("area_ha", 0)), 1),
                "soil_type": str(row.get("soil_type", "")),
                "predicted_yield_kg_ha": round(float(preds[0]), 1),
                "lower_bound": round(float(lower[0]), 1),
                "upper_bound": round(float(upper[0]), 1),
                "drift_warning": drift_result["drift_warning"],
                "drift_level": drift_result["drift_level"],
                "stale_features": live.get("stale_features", False),
                "error": None,
            }
        except Exception as exc:
            log.warning("Prediction failed for %s: %s", field_id, exc)
            return {
                "field_id": field_id,
                "name": str(row.get("name", field_id)),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "crop_type": str(row.get("crop_type", "")),
                "region": str(row.get("region", "")),
                "area_ha": round(float(row.get("area_ha", 0)), 1),
                "soil_type": str(row.get("soil_type", "")),
                "predicted_yield_kg_ha": None,
                "lower_bound": None,
                "upper_bound": None,
                "drift_warning": False,
                "drift_level": "none",
                "stale_features": False,
                "error": str(exc),
            }


@app.get("/fields")
async def bulk_fields():
    """Return all fields with live predictions — powers the map UI."""
    if _FIELDS_DF.empty:
        raise HTTPException(status_code=503, detail="Fields not loaded.")
    if not model_module.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    tasks = [
        _predict_one(fid, row)
        for fid, row in _FIELDS_DF.iterrows()
    ]
    results = await asyncio.gather(*tasks)
    return {"fields": list(results), "model_version": model_module.model_version()}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def map_ui():
    html = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agri Yield — UK Field Intelligence</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300..700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:        #0b0f17;
      --surface:   #111827;
      --surface2:  #1a2233;
      --border:    rgba(255,255,255,0.07);
      --text:      #e2e8f0;
      --muted:     #64748b;
      --faint:     #334155;
      --green:     #4ade80;
      --green-d:   #16a34a;
      --yellow:    #facc15;
      --orange:    #fb923c;
      --red:       #f87171;
      --purple:    #a78bfa;
      --blue:      #38bdf8;
      --teal:      #2dd4bf;
      --radius:    10px;
      --transition: 200ms cubic-bezier(0.16,1,0.3,1);
    }
    html, body { height: 100%; font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); overflow: hidden; }

    /* ── LAYOUT ── */
    #app { display: grid; grid-template-rows: 56px 1fr; height: 100dvh; }
    #map-wrap { position: relative; overflow: hidden; }
    #map { width: 100%; height: 100%; }

    /* ── HEADER ── */
    header {
      display: flex; align-items: center; gap: 14px;
      padding: 0 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      z-index: 1000;
    }
    .logo { display: flex; align-items: center; gap: 9px; }
    .logo-mark {
      width: 30px; height: 30px; border-radius: 8px;
      background: linear-gradient(135deg, #14532d, #166534);
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .logo-mark svg { color: var(--green); }
    .logo-name { font-size: 0.9rem; font-weight: 650; letter-spacing: -0.02em; color: #f1f5f9; }
    .logo-tag  { font-size: 0.68rem; color: var(--muted); margin-top: 1px; }

    /* Stats strip */
    #stats-strip {
      display: flex; gap: 2px; margin-left: auto; align-items: center;
    }
    .stat {
      display: flex; flex-direction: column; align-items: flex-end;
      padding: 0 12px; border-right: 1px solid var(--border);
    }
    .stat:last-child { border-right: none; }
    .stat-val { font-size: 0.85rem; font-weight: 600; color: #f1f5f9; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .stat-lbl { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin-top: 1px; }
    .stat-val.green { color: var(--green); }
    .stat-val.yellow { color: var(--yellow); }
    .stat-val.red { color: var(--red); }

    /* Live pill */
    .live-pill {
      display: flex; align-items: center; gap: 6px;
      padding: 5px 12px; border-radius: 999px;
      background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.2);
      font-size: 0.7rem; font-weight: 500; color: var(--green);
      margin-left: 12px;
    }
    .live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: pulse 2s ease infinite; }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.3)} }

    /* ── FILTER BAR ── */
    #filter-bar {
      position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
      z-index: 900;
      display: flex; gap: 6px; align-items: center;
      background: var(--surface); border: 1px solid var(--border);
      padding: 6px 10px; border-radius: 999px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }
    .filter-btn {
      padding: 4px 12px; border-radius: 999px; font-size: 0.72rem; font-weight: 500;
      border: 1px solid transparent; cursor: pointer; transition: all var(--transition);
      color: var(--muted); background: transparent;
    }
    .filter-btn:hover { color: var(--text); background: var(--surface2); }
    .filter-btn.active { color: var(--bg); font-weight: 600; }
    .filter-btn[data-crop="all"].active        { background: var(--text);   border-color: var(--text); }
    .filter-btn[data-crop="winter_wheat"].active,
    .filter-btn[data-crop="spring_wheat"].active  { background: var(--green);  border-color: var(--green); }
    .filter-btn[data-crop="winter_barley"].active,
    .filter-btn[data-crop="spring_barley"].active { background: var(--yellow); border-color: var(--yellow); }
    .filter-btn[data-crop="oilseed_rape"].active  { background: var(--orange); border-color: var(--orange); }
    .filter-btn[data-crop="sugar_beet"].active    { background: var(--blue);   border-color: var(--blue); }
    .filter-divider { width: 1px; height: 18px; background: var(--border); }
    .filter-btn[data-view="yield"].active  { background: var(--teal);   border-color: var(--teal); }
    .filter-btn[data-view="crop"].active   { background: var(--purple); border-color: var(--purple); }
    .filter-btn[data-view="drift"].active  { background: var(--red);    border-color: var(--red); }

    /* ── SIDE PANEL ── */
    #panel {
      position: absolute; top: 12px; right: 12px; bottom: 12px;
      width: 310px; z-index: 900;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); overflow: hidden;
      display: flex; flex-direction: column;
      box-shadow: 0 8px 40px rgba(0,0,0,0.6);
      transform: translateX(340px); transition: transform 0.3s cubic-bezier(0.16,1,0.3,1);
    }
    #panel.open { transform: translateX(0); }

    .panel-header {
      padding: 14px 16px 12px;
      border-bottom: 1px solid var(--border);
      display: flex; justify-content: space-between; align-items: flex-start;
      flex-shrink: 0;
    }
    .p-field-id  { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--green); }
    .p-name      { font-size: 1.05rem; font-weight: 650; color: #f1f5f9; margin-top: 2px; letter-spacing: -0.02em; }
    .p-crop-tag  { font-size: 0.7rem; color: var(--muted); margin-top: 3px; text-transform: capitalize; }
    .panel-close {
      background: none; border: none; color: var(--muted); cursor: pointer;
      width: 26px; height: 26px; border-radius: 6px; display: flex; align-items: center; justify-content: center;
      font-size: 0.9rem; transition: all var(--transition); flex-shrink: 0;
    }
    .panel-close:hover { background: var(--surface2); color: var(--text); }

    .panel-meta {
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 1px; background: var(--border); flex-shrink: 0;
    }
    .meta-cell {
      background: var(--surface); padding: 10px 14px;
      display: flex; flex-direction: column; gap: 3px;
    }
    .meta-lbl { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .meta-val  { font-size: 0.82rem; font-weight: 500; color: #cbd5e1; }

    /* Soil badge */
    .soil-badge {
      display: inline-block; padding: 2px 8px; border-radius: 999px;
      font-size: 0.65rem; font-weight: 500; text-transform: capitalize;
      background: rgba(165,180,252,0.1); color: #a5b4fc; border: 1px solid rgba(165,180,252,0.2);
    }

    /* Yield block */
    #panel-yield { padding: 16px; flex-shrink: 0; border-bottom: 1px solid var(--border); }
    .yield-main  { display: flex; align-items: flex-end; gap: 6px; }
    .yield-num   { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
    .yield-unit  { font-size: 0.75rem; color: var(--muted); padding-bottom: 5px; }
    .yield-ci    { font-size: 0.72rem; color: var(--muted); margin-top: 5px; }
    .yield-bar-track { margin-top: 10px; background: var(--surface2); border-radius: 999px; height: 5px; overflow: hidden; }
    .yield-bar-fill  { height: 100%; border-radius: 999px; transition: width 0.8s cubic-bezier(0.16,1,0.3,1); }

    /* Weather strip */
    #panel-weather {
      padding: 12px 16px; flex-shrink: 0;
      border-bottom: 1px solid var(--border);
      display: grid; grid-template-columns: repeat(3,1fr); gap: 8px;
    }
    .wx-cell { display: flex; flex-direction: column; align-items: center; gap: 4px; }
    .wx-icon { font-size: 1.2rem; }
    .wx-val  { font-size: 0.78rem; font-weight: 600; color: #f1f5f9; }
    .wx-lbl  { font-size: 0.6rem; color: var(--muted); text-align: center; }

    /* Drift / badges */
    #panel-badges { padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 6px; flex-shrink: 0; }
    .badge {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 3px 10px; border-radius: 999px;
      font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
    }
    .badge-drift-none  { background: rgba(74,222,128,0.1);  color: var(--green);  border: 1px solid rgba(74,222,128,0.25); }
    .badge-drift-low   { background: rgba(250,204,21,0.1);  color: var(--yellow); border: 1px solid rgba(250,204,21,0.25); }
    .badge-drift-high  { background: rgba(248,113,113,0.1); color: var(--red);    border: 1px solid rgba(248,113,113,0.25); }
    .badge-stale       { background: rgba(251,146,60,0.1);  color: var(--orange); border: 1px solid rgba(251,146,60,0.25); }
    .badge-model       { background: rgba(45,212,191,0.08); color: var(--teal);   border: 1px solid rgba(45,212,191,0.2); }

    /* Loading / skeleton */
    #panel-loading {
      flex: 1; display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 12px; color: var(--muted); font-size: 0.8rem;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner { width: 24px; height: 24px; border: 2px solid var(--surface2); border-top-color: var(--green); border-radius: 50%; animation: spin 0.7s linear infinite; }
    @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
    .skel {
      border-radius: 6px; height: 12px;
      background: linear-gradient(90deg, var(--surface2) 25%, var(--faint) 50%, var(--surface2) 75%);
      background-size:200% 100%; animation:shimmer 1.4s ease-in-out infinite;
    }

    /* ── LEGEND ── */
    #legend {
      position: absolute; bottom: 20px; left: 14px; z-index: 900;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 12px 14px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
      min-width: 150px;
    }
    #legend-title { font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }
    .leg-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: 0.72rem; color: #94a3b8; }
    .leg-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    /* yield gradient legend */
    .leg-gradient { width: 100%; height: 6px; border-radius: 999px; margin: 6px 0 4px; background: linear-gradient(90deg, #f87171, #facc15, #4ade80); }
    .leg-gradient-labels { display: flex; justify-content: space-between; font-size: 0.6rem; color: var(--muted); }

    /* ── LOADING OVERLAY ── */
    #overlay {
      position: absolute; inset: 0; z-index: 9999;
      background: var(--bg);
      display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px;
      transition: opacity 0.6s ease;
    }
    #overlay.fade-out { opacity: 0; pointer-events: none; }
    .overlay-logo { display: flex; flex-direction: column; align-items: center; gap: 10px; }
    .overlay-title { font-size: 1.1rem; font-weight: 650; letter-spacing: -0.02em; color: #f1f5f9; }
    .overlay-sub   { font-size: 0.75rem; color: var(--muted); }
    #progress-bar-track { width: 200px; height: 3px; background: var(--surface2); border-radius: 999px; overflow: hidden; margin-top: 8px; }
    #progress-bar-fill  { height: 100%; width: 0%; background: var(--green); border-radius: 999px; transition: width 0.4s ease; }
    #overlay-status { font-size: 0.72rem; color: var(--muted); min-height: 1em; }

    /* ── LEAFLET DARK THEME ── */
    .leaflet-container { background: var(--bg) !important; }
    .leaflet-control-zoom a { background: var(--surface) !important; color: #94a3b8 !important; border-color: var(--border) !important; }
    .leaflet-control-zoom a:hover { background: var(--surface2) !important; color: var(--text) !important; }
    .leaflet-control-attribution { background: rgba(17,24,39,0.7) !important; color: #334155 !important; font-size: 0.6rem !important; }
    .leaflet-control-attribution a { color: var(--green) !important; }
    .leaflet-popup-content-wrapper { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--text) !important; box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important; }
    .leaflet-popup-tip { background: var(--surface) !important; }
    .leaflet-popup-content { margin: 10px 14px !important; font-size: 0.78rem !important; line-height: 1.5; }

    @media (max-width: 600px) {
      #panel { left: 0; right: 0; top: auto; bottom: 0; width: 100%; height: 65vh; border-radius: var(--radius) var(--radius) 0 0; transform: translateY(100%); }
      #panel.open { transform: translateY(0); }
      #stats-strip .stat:nth-child(n+4) { display: none; }
      #filter-bar { top: 8px; gap: 4px; padding: 4px 8px; }
      .filter-btn { padding: 3px 9px; font-size: 0.68rem; }
    }
  </style>
</head>
<body>
<div id="app">
  <header>
    <div class="logo">
      <div class="logo-mark">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M8 2C8 2 4 5 4 9.5C4 11.99 5.79 14 8 14C10.21 14 12 11.99 12 9.5C12 5 8 2 8 2Z" fill="currentColor" opacity="0.9"/>
          <path d="M8 14V7" stroke="#14532d" stroke-width="1.2" stroke-linecap="round"/>
          <path d="M8 10L6 8" stroke="#14532d" stroke-width="1" stroke-linecap="round"/>
          <path d="M8 8.5L10 6.5" stroke="#14532d" stroke-width="1" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <div class="logo-name">Agri Yield</div>
        <div class="logo-tag">UK Field Intelligence</div>
      </div>
    </div>
    <div id="stats-strip">
      <div class="stat"><span class="stat-val" id="s-fields">—</span><span class="stat-lbl">Fields</span></div>
      <div class="stat"><span class="stat-val green" id="s-avg">—</span><span class="stat-lbl">Avg yield kg/ha</span></div>
      <div class="stat"><span class="stat-val green" id="s-best-val">—</span><span class="stat-lbl" id="s-best-lbl">Top field</span></div>
      <div class="stat"><span class="stat-val red" id="s-drift">—</span><span class="stat-lbl">Drift warnings</span></div>
      <div class="stat"><span class="stat-val" id="s-model">—</span><span class="stat-lbl">Model</span></div>
    </div>
    <div class="live-pill"><div class="live-dot"></div>Live</div>
  </header>

  <div id="map-wrap">
    <div id="map"></div>

    <!-- Filter bar -->
    <div id="filter-bar">
      <button class="filter-btn active" data-crop="all">All</button>
      <button class="filter-btn" data-crop="winter_wheat">Wheat</button>
      <button class="filter-btn" data-crop="winter_barley">Barley</button>
      <button class="filter-btn" data-crop="oilseed_rape">OSR</button>
      <button class="filter-btn" data-crop="sugar_beet">Sugar Beet</button>
      <div class="filter-divider"></div>
      <button class="filter-btn active" data-view="yield">Yield</button>
      <button class="filter-btn" data-view="crop">Crop</button>
      <button class="filter-btn" data-view="drift">Drift</button>
    </div>

    <!-- Side panel -->
    <div id="panel">
      <div class="panel-header">
        <div>
          <div class="p-field-id" id="p-id">—</div>
          <div class="p-name"    id="p-name">—</div>
          <div class="p-crop-tag" id="p-crop">—</div>
        </div>
        <button class="panel-close" onclick="closePanel()" aria-label="Close panel">✕</button>
      </div>
      <div class="panel-meta">
        <div class="meta-cell"><span class="meta-lbl">Region</span><span class="meta-val" id="p-region">—</span></div>
        <div class="meta-cell"><span class="meta-lbl">Area</span><span class="meta-val" id="p-area">—</span></div>
        <div class="meta-cell"><span class="meta-lbl">Coordinates</span><span class="meta-val" id="p-coords">—</span></div>
        <div class="meta-cell"><span class="meta-lbl">Soil Type</span><span class="meta-val" id="p-soil">—</span></div>
      </div>
      <div id="panel-yield">
        <div class="yield-main">
          <div class="yield-num" id="p-yield">—</div>
          <div class="yield-unit">kg / ha</div>
        </div>
        <div class="yield-ci" id="p-ci">—</div>
        <div class="yield-bar-track"><div class="yield-bar-fill" id="p-bar" style="width:0%"></div></div>
      </div>
      <div id="panel-weather">
        <div class="wx-cell"><div class="wx-icon">&#127777;</div><div class="wx-val" id="wx-temp">—</div><div class="wx-lbl">Temperature</div></div>
        <div class="wx-cell"><div class="wx-icon">&#127783;</div><div class="wx-val" id="wx-rain">—</div><div class="wx-lbl">Precipitation</div></div>
        <div class="wx-cell"><div class="wx-icon">&#9728;</div><div class="wx-val" id="wx-rad">—</div><div class="wx-lbl">Solar Rad.</div></div>
      </div>
      <div id="panel-badges"></div>
    </div>

    <!-- Legend -->
    <div id="legend">
      <div id="legend-title">Colour by yield</div>
      <div id="legend-body">
        <div class="leg-gradient"></div>
        <div class="leg-gradient-labels"><span>Low</span><span>High</span></div>
      </div>
    </div>

    <!-- Loading overlay -->
    <div id="overlay">
      <div class="overlay-logo">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
          <rect width="40" height="40" rx="10" fill="#14532d"/>
          <path d="M20 7C20 7 11 13 11 22C11 27.52 15.03 32 20 32C24.97 32 29 27.52 29 22C29 13 20 7 20 7Z" fill="#4ade80"/>
          <path d="M20 32V17" stroke="#14532d" stroke-width="2" stroke-linecap="round"/>
          <path d="M20 24L15 19" stroke="#14532d" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M20 20L25 15" stroke="#14532d" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <div class="overlay-title">Agri Yield Intelligence</div>
        <div class="overlay-sub" id="overlay-status">Connecting to model\u2026</div>
        <div id="progress-bar-track"><div id="progress-bar-fill"></div></div>
      </div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(function () {
  // ── Colour helpers ──
  const CROP_COLORS = {
    winter_wheat:   '#4ade80',
    spring_wheat:   '#86efac',
    winter_barley:  '#facc15',
    spring_barley:  '#fde047',
    oilseed_rape:   '#fb923c',
    sugar_beet:     '#38bdf8',
    potato:         '#a78bfa',
  };
  const CROP_LABELS = {
    winter_wheat:  'Winter Wheat',  spring_wheat:  'Spring Wheat',
    winter_barley: 'Winter Barley', spring_barley: 'Spring Barley',
    oilseed_rape:  'Oilseed Rape',  sugar_beet:    'Sugar Beet',
    potato:        'Potato',
  };
  function cropColor(c) { return CROP_COLORS[c] || '#94a3b8'; }
  function lerpColor(t) {
    // red → yellow → green gradient for yield
    if (t < 0.5) {
      const r = 248, g = Math.round(113 + (204 - 113) * (t / 0.5)), b = 113;
      return `rgb(${r},${g},${b})`;
    } else {
      const r = Math.round(250 + (74 - 250) * ((t - 0.5) / 0.5));
      const g = Math.round(204 + (222 - 204) * ((t - 0.5) / 0.5));
      const b = Math.round(21 + (128 - 21) * ((t - 0.5) / 0.5));
      return `rgb(${r},${g},${b})`;
    }
  }
  function markerColor(field, viewMode, minY, maxY) {
    if (viewMode === 'yield') {
      if (field.predicted_yield_kg_ha == null) return '#334155';
      const t = (field.predicted_yield_kg_ha - minY) / Math.max(maxY - minY, 1);
      return lerpColor(Math.max(0, Math.min(1, t)));
    }
    if (viewMode === 'crop')  return cropColor(field.crop_type);
    if (viewMode === 'drift') {
      if (field.drift_level === 'high') return '#f87171';
      if (field.drift_level === 'low')  return '#facc15';
      return '#4ade80';
    }
    return '#94a3b8';
  }
  function makeIcon(color, radius) {
    const r = radius || 10;
    const s = r * 2 + 4;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}" viewBox="0 0 ${s} ${s}">
      <circle cx="${s/2}" cy="${s/2}" r="${r}" fill="${color}" fill-opacity="0.22" stroke="${color}" stroke-width="2"/>
      <circle cx="${s/2}" cy="${s/2}" r="${r*0.42}" fill="${color}"/>
    </svg>`;
    return L.divIcon({ html: svg, className: '', iconSize: [s, s], iconAnchor: [s/2, s/2], popupAnchor: [0, -s/2] });
  }

  // ── Map init ──
  const map = L.map('map', { center: [54, -2.5], zoom: 6, zoomControl: true });
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://openstreetmap.org">OSM</a>',
    subdomains: 'abcd', maxZoom: 19,
  }).addTo(map);

  // ── State ──
  let allFields = [];
  let markers = [];
  let activeCrop = 'all';
  let activeView = 'yield';
  let minYield = 0, maxYield = 1;
  let activeField = null;

  // ── Progress bar ──
  function setProgress(pct, msg) {
    document.getElementById('progress-bar-fill').style.width = pct + '%';
    document.getElementById('overlay-status').textContent = msg || '';
  }

  // ── Fetch all fields ──
  setProgress(10, 'Loading field data\u2026');
  fetch('/fields')
    .then(r => r.json())
    .then(data => {
      setProgress(70, 'Rendering map\u2026');
      allFields = data.fields;
      document.getElementById('s-model').textContent = data.model_version || '\u2014';
      computeStats();
      buildMarkers();
      updateLegend();
      setProgress(100, 'Ready');
      setTimeout(() => {
        const ov = document.getElementById('overlay');
        ov.classList.add('fade-out');
        setTimeout(() => ov.remove(), 700);
      }, 300);
    })
    .catch(err => {
      document.getElementById('overlay-status').textContent = '\u26a0 Failed to load: ' + err.message;
    });

  // ── Stats ──
  function computeStats() {
    const yields = allFields.filter(f => f.predicted_yield_kg_ha != null).map(f => f.predicted_yield_kg_ha);
    if (!yields.length) return;
    minYield = Math.min(...yields);
    maxYield = Math.max(...yields);
    const avg = yields.reduce((a, b) => a + b, 0) / yields.length;
    const best = allFields.reduce((a, b) => (b.predicted_yield_kg_ha || 0) > (a.predicted_yield_kg_ha || 0) ? b : a);
    const driftCount = allFields.filter(f => f.drift_warning).length;
    document.getElementById('s-fields').textContent = allFields.length;
    document.getElementById('s-avg').textContent = avg.toLocaleString(undefined, { maximumFractionDigits: 0 });
    document.getElementById('s-best-val').textContent = (best.predicted_yield_kg_ha || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });
    document.getElementById('s-best-lbl').textContent = best.name || 'Top field';
    const driftEl = document.getElementById('s-drift');
    driftEl.textContent = driftCount;
    driftEl.className = 'stat-val ' + (driftCount > 5 ? 'red' : driftCount > 0 ? 'yellow' : 'green');
  }

  // ── Build markers ──
  function buildMarkers() {
    markers.forEach(m => map.removeLayer(m.layer));
    markers = [];
    const filtered = activeCrop === 'all'
      ? allFields
      : allFields.filter(f => f.crop_type === activeCrop || f.crop_type.includes(activeCrop.replace('winter_','').replace('spring_','')));
    filtered.forEach(f => {
      const color = markerColor(f, activeView, minYield, maxYield);
      const radius = Math.max(8, Math.min(15, 8 + (f.area_ha / 30)));
      const layer = L.marker([f.lat, f.lon], { icon: makeIcon(color, radius) }).addTo(map);
      layer.on('click', () => openPanel(f));
      layer.bindTooltip(`<strong>${f.name}</strong><br>${(CROP_LABELS[f.crop_type] || f.crop_type)} &mdash; ${f.area_ha} ha`, { direction: 'top', offset: [0, -10] });
      markers.push({ layer, field: f });
    });
  }

  // ── Legend ──
  function updateLegend() {
    const title = document.getElementById('legend-title');
    const body  = document.getElementById('legend-body');
    if (activeView === 'yield') {
      title.textContent = 'Colour by yield';
      body.innerHTML = `<div class="leg-gradient"></div><div class="leg-gradient-labels"><span>${Math.round(minYield).toLocaleString()}</span><span>${Math.round(maxYield).toLocaleString()} kg/ha</span></div>`;
    } else if (activeView === 'crop') {
      title.textContent = 'Crop type';
      const crops = [...new Set(allFields.map(f => f.crop_type))];
      body.innerHTML = crops.map(c => `<div class="leg-row"><div class="leg-dot" style="background:${cropColor(c)}"></div>${CROP_LABELS[c] || c}</div>`).join('');
    } else {
      title.textContent = 'Drift level';
      body.innerHTML = [
        ['#4ade80','No drift'],['#facc15','Low drift'],['#f87171','High drift']
      ].map(([c,l]) => `<div class="leg-row"><div class="leg-dot" style="background:${c}"></div>${l}</div>`).join('');
    }
  }

  // ── Filters ──
  document.querySelectorAll('[data-crop]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-crop]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeCrop = btn.dataset.crop;
      buildMarkers();
    });
  });
  document.querySelectorAll('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeView = btn.dataset.view;
      buildMarkers();
      updateLegend();
    });
  });

  // ── Panel ──
  function openPanel(f) {
    activeField = f;
    document.getElementById('p-id').textContent = f.field_id;
    document.getElementById('p-name').textContent = f.name || f.field_id;
    document.getElementById('p-crop').textContent = (CROP_LABELS[f.crop_type] || f.crop_type).replace(/_/g,' ');
    document.getElementById('p-region').textContent = f.region || '\u2014';
    document.getElementById('p-area').textContent = f.area_ha + ' ha';
    document.getElementById('p-coords').textContent = f.lat.toFixed(3) + ', ' + f.lon.toFixed(3);
    const soil = (f.soil_type || '').replace(/_/g,' ');
    document.getElementById('p-soil').innerHTML = soil ? `<span class="soil-badge">${soil}</span>` : '\u2014';

    // Yield
    const y = f.predicted_yield_kg_ha;
    const color = y != null ? markerColor(f, 'yield', minYield, maxYield) : '#94a3b8';
    const pct = y != null ? Math.max(4, Math.round(((y - minYield) / Math.max(maxYield - minYield, 1)) * 100)) : 0;
    document.getElementById('p-yield').textContent = y != null ? y.toLocaleString(undefined,{maximumFractionDigits:0}) : '\u2014';
    document.getElementById('p-yield').style.color = color;
    document.getElementById('p-ci').textContent = f.lower_bound != null
      ? `80% CI: ${f.lower_bound.toLocaleString(undefined,{maximumFractionDigits:0})} \u2013 ${f.upper_bound.toLocaleString(undefined,{maximumFractionDigits:0})} kg/ha`
      : '';
    const bar = document.getElementById('p-bar');
    bar.style.background = color;
    setTimeout(() => { bar.style.width = pct + '%'; }, 50);

    // Weather placeholders
    document.getElementById('wx-temp').textContent = '\u2014';
    document.getElementById('wx-rain').textContent = '\u2014';
    document.getElementById('wx-rad').textContent  = '\u2014';

    // Badges
    const driftClass = f.drift_level === 'high' ? 'badge-drift-high' : f.drift_level === 'low' ? 'badge-drift-low' : 'badge-drift-none';
    const driftIcon  = f.drift_level === 'high' ? '\u26a0\ufe0f' : f.drift_level === 'low' ? '\u25b3' : '\u2713';
    const driftLabel = f.drift_warning ? `Drift: ${f.drift_level}` : 'No drift';
    let badges = `<span class="badge ${driftClass}">${driftIcon} ${driftLabel}</span>`;
    if (f.stale_features) badges += `<span class="badge badge-stale">\u23f0 Stale features</span>`;
    badges += `<span class="badge badge-model">\u2b22 ${document.getElementById('s-model').textContent}</span>`;
    document.getElementById('panel-badges').innerHTML = badges;

    // Fetch fresh weather for the wx strip
    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ field_id: f.field_id }),
    }).then(r => r.json()).then(d => {
      if (d.stale_features) {
        document.getElementById('wx-temp').textContent = 'Stale';
        document.getElementById('wx-rain').textContent = 'Stale';
        document.getElementById('wx-rad').textContent  = 'Stale';
      } else {
        document.getElementById('wx-temp').textContent = 'Live';
        document.getElementById('wx-rain').textContent = 'Live';
        document.getElementById('wx-rad').textContent  = 'Live';
      }
    }).catch(() => {});

    document.getElementById('panel').classList.add('open');
    map.panTo([f.lat, f.lon], { animate: true, duration: 0.4 });
  }

  window.closePanel = function () {
    document.getElementById('panel').classList.remove('open');
    document.getElementById('p-bar').style.width = '0%';
  };

  // Close panel on map click
  map.on('click', (e) => {
    if (!e.originalEvent.target.closest) return;
    closePanel();
  });
})();
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
