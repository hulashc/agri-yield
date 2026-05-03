# Phase 6 — Public Frontend + Real Data Integration

## What We Are Building

A public-facing Next.js web app that calls the existing FastAPI backend, displays
per-field yield predictions on an interactive UK map, and shows live monitoring
status (drift, freshness, latency). The backend gains real weather data from
NASA POWER (historical) and Open-Meteo (live).

Anyone — farmer, recruiter, journalist — can open the URL, click a field, and
get a yield prediction with a confidence range and a plain-English explanation
of what the model used to make it.

---

## Repository Structure After Phase 6

```
agri-yield/
├── serving/              # existing FastAPI backend (unchanged API contract)
│   ├── app.py
│   ├── feast_client.py   # UPDATED — real feature fetch
│   ├── schemas.py
│   └── ...
├── ingestion/
│   ├── nasa_power.py     # NEW — historical weather fetch (NASA POWER API)
│   └── open_meteo.py     # NEW — live weather fetch (Open-Meteo API)
├── features/             # existing Feast feature store
│   └── feature_store.yaml
├── web/                  # NEW — Next.js frontend
│   ├── app/
│   │   ├── page.tsx          # landing page with map
│   │   ├── layout.tsx
│   │   └── api/
│   │       └── predict/
│   │           └── route.ts  # server-side proxy to FastAPI
│   ├── components/
│   │   ├── Map.tsx           # Leaflet map with field markers
│   │   ├── PredictionCard.tsx
│   │   ├── DriftBadge.tsx
│   │   └── MonitoringBar.tsx
│   ├── lib/
│   │   └── api.ts            # typed fetch wrappers
│   ├── public/
│   │   └── fields.geojson    # UK demo field polygons
│   ├── package.json
│   └── next.config.ts
├── docs/
│   └── phase-6-frontend.md  # this file
└── pyproject.toml
```

---

## Step 1 — Wire Real Data into the Backend

### 1a. NASA POWER — Historical Feature Materialisation

File: `ingestion/nasa_power.py`

```python
import httpx
import pandas as pd

FIELDS = {
    "field-lincs-001": {"lat": 53.15, "lon": -0.35},   # Lincolnshire arable
    "field-lincs-002": {"lat": 53.22, "lon": -0.41},
    "field-yorks-001": {"lat": 53.88, "lon": -1.12},   # Yorkshire Wolds
    "field-norfolk-001": {"lat": 52.65, "lon":  0.88}, # Norfolk Broads edge
    "field-cambs-001": {"lat": 52.30, "lon":  0.10},   # Cambridgeshire fens
}

NASA_PARAMS = "PRECTOTCORR,T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN,RH2M,WS2M"

def fetch_nasa_power(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={NASA_PARAMS}&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON"
    )
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]
    return pd.DataFrame(data)
```

Run once to build the historical feature table used for training and PSI baselines:

```bash
uv run python -m ingestion.nasa_power
```

---

### 1b. Open-Meteo — Live Features for /predict

File: `ingestion/open_meteo.py`

```python
import httpx

def fetch_live_features(lat: float, lon: float) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,"
        "et0_fao_evapotranspiration,shortwave_radiation_sum"
        "&forecast_days=1&timezone=Europe/London"
    )
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    d = resp.json()["daily"]
    return {
        "precipitation_sum":        d["precipitation_sum"][0],
        "temperature_max":          d["temperature_2m_max"][0],
        "temperature_min":          d["temperature_2m_min"][0],
        "et0_evapotranspiration":   d["et0_fao_evapotranspiration"][0],
        "shortwave_radiation_sum":  d["shortwave_radiation_sum"][0],
    }
```

Update `serving/feast_client.py` to call `fetch_live_features` when the
field_id is one of the real fields, falling back to Feast for others.

---

## Step 2 — Add /fields Endpoint to FastAPI

Add to `serving/app.py`:

```python
FIELDS = {
    "field-lincs-001":   {"lat": 53.15, "lon": -0.35, "name": "Lincolnshire North"},
    "field-lincs-002":   {"lat": 53.22, "lon": -0.41, "name": "Lincolnshire South"},
    "field-yorks-001":   {"lat": 53.88, "lon": -1.12, "name": "Yorkshire Wolds"},
    "field-norfolk-001": {"lat": 52.65, "lon":  0.88, "name": "Norfolk"},
    "field-cambs-001":   {"lat": 52.30, "lon":  0.10, "name": "Cambridgeshire Fens"},
}

@app.get("/fields")
def list_fields():
    return FIELDS
```

This lets the frontend load field locations without hardcoding them.

---

## Step 3 — Build the Next.js Frontend

### Setup

```bash
cd agri-yield
npx create-next-app@latest web --typescript --tailwind --app --no-src-dir
cd web
npm install leaflet react-leaflet @types/leaflet
```

### Map Component (`web/components/Map.tsx`)

```tsx
"use client";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

type Field = {
  id: string;
  lat: number;
  lon: number;
  name: string;
  prediction?: number;
  lower?: number;
  upper?: number;
};

export default function Map({ fields }: { fields: Field[] }) {
  return (
    <MapContainer center={[52.8, -0.5]} zoom={7} className="h-[600px] w-full rounded-xl">
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="© OpenStreetMap contributors"
      />
      {fields.map((f) => (
        <CircleMarker
          key={f.id}
          center={[f.lat, f.lon]}
          radius={12}
          pathOptions={{ color: "#16a34a", fillColor: "#22c55e", fillOpacity: 0.8 }}
        >
          <Popup>
            <strong>{f.name}</strong><br />
            {f.prediction
              ? `${f.prediction.toLocaleString()} kg/ha (${f.lower?.toLocaleString()}–${f.upper?.toLocaleString()})`
              : "Click to load prediction"}
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
```

### Prediction Card (`web/components/PredictionCard.tsx`)

Shows the result after a field is clicked:
- Predicted yield in large type
- Confidence band as a range bar
- Drift warning badge (red if PSI > 0.2)
- Model version + last updated timestamp

### Landing Page (`web/app/page.tsx`)

```tsx
import dynamic from "next/dynamic";

const Map = dynamic(() => import("../components/Map"), { ssr: false });

export default async function Home() {
  const res = await fetch(`${process.env.API_URL}/fields`);
  const fields = await res.json();

  return (
    <main className="max-w-5xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-2">UK Crop Yield Forecast</h1>
      <p className="text-gray-500 mb-6">
        Real-time yield predictions for UK arable fields using satellite weather
        data and a gradient-boosted ML model. Click a field to see today's forecast.
      </p>
      <Map fields={Object.entries(fields).map(([id, v]: any) => ({ id, ...v }))} />
    </main>
  );
}
```

---

## Step 4 — Deployment

### Backend — Railway or Fly.io (free tier)

```bash
# Railway (recommended — free, no credit card for small apps)
npm install -g @railway/cli
railway login
railway init
railway up
```

Set environment variable: `PORT=8000`

### Frontend — Vercel (free)

```bash
cd web
vercel deploy --prod
```

Set environment variable in Vercel dashboard:
```
API_URL=https://your-railway-app.up.railway.app
```

### CORS — add to `serving/app.py`

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-app.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Step 5 — What the Grafana Dashboard Shows After This

| Panel | Before | After |
|---|---|---|
| Prediction Request Rate | Only /health and /metrics | Real /predict calls from the web UI |
| Prediction Latency | ~100ms (no real work) | Real inference latency with feature fetch |
| Feature PSI (Drift Score) | No data | Real PSI vs NASA POWER baseline |
| Feature Freshness | No data | Actual Open-Meteo fetch timestamp |

---

## What a Recruiter Sees

1. **Live URL** — a real website they can open and interact with
2. **GitHub repo** — clean structure, phased commits, model card, monitoring docs
3. **Grafana screenshot** — all 4 panels showing live data
4. **The pitch** — "I built an end-to-end MLOps system for agricultural yield
   prediction. It ingests real satellite weather data, serves predictions with
   uncertainty bounds, monitors for data drift using PSI, and triggers automated
   retraining. The public frontend lets anyone query a UK field and get a forecast."

---

## Build Order

- [ ] `ingestion/nasa_power.py` — fetch and store historical features
- [ ] `ingestion/open_meteo.py` — live feature fetch
- [ ] Update `serving/feast_client.py` — wire Open-Meteo into /predict
- [ ] Add `/fields` endpoint to `serving/app.py`
- [ ] Add CORS middleware to `serving/app.py`
- [ ] `npx create-next-app web` — scaffold frontend
- [ ] Build `Map.tsx`, `PredictionCard.tsx`, `MonitoringBar.tsx`
- [ ] Build `app/page.tsx` — landing page
- [ ] Deploy backend to Railway
- [ ] Deploy frontend to Vercel
- [ ] Update Grafana dashboard — confirm all 4 panels show live data
- [ ] Screenshot dashboard + record a short Loom walkthrough for portfolio
