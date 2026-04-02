# data/simulation.py  — PRODUCTION VERSION
# =============================================================================
# AI Farm Management Agent System v2 — Real API Data Layer
# =============================================================================
#
# This file is a drop-in replacement for the simulation-based simulation.py.
# All four functions return the IDENTICAL dict structure as the simulation
# version — no changes to any agent, graph, or tool file are needed.
#
# SETUP:
#   1. pip install requests earthengine-api google-auth
#   2. Fill in your .env file (copy from .env.example)
#   3. Replace data/simulation.py with this file
#   4. Run: python main.py
#
# ENVIRONMENT VARIABLES REQUIRED (set in .env):
#   OPENWEATHERMAP_API_KEY   — weather + 7-day forecast
#   SOIL_SENSOR_API_URL      — your Sentek/Stevens gateway base URL
#   SOIL_SENSOR_API_KEY      — gateway authentication token
#   CME_API_KEY              — CME Group DataMine for commodity prices
#   GEE_PROJECT_ID           — Google Earth Engine project for NDVI
#
# PARTIAL ADOPTION:
#   You do not need to connect all four APIs at once.
#   Any function that cannot find its env var will fall back to
#   the simulation data automatically and print a warning.
#   This lets you connect APIs one at a time.
#
# ZONE CONFIGURATION:
#   Edit the ZONES list below to match your actual farm zones,
#   crop types, and GPS coordinates.
# =============================================================================

import os
import json
import logging
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

# =============================================================================
# ZONE DEFINITIONS
# Edit this list to match your actual farm layout.
# lat/lon are used for soil sensor queries and GEE NDVI pulls.
# =============================================================================

ZONES = [
    {
        "zone_id":    "Z-NORTH",
        "crop":       "wheat",
        "area_acres": 40.0,
        "lat":        36.7783,   # replace with your GPS coordinates
        "lon":        -119.4179,
    },
    {
        "zone_id":    "Z-SOUTH",
        "crop":       "corn",
        "area_acres": 45.0,
        "lat":        36.7650,
        "lon":        -119.4100,
    },
    {
        "zone_id":    "Z-EAST",
        "crop":       "soybeans",
        "area_acres": 35.0,
        "lat":        36.7700,
        "lon":        -119.4000,
    },
    {
        "zone_id":    "Z-WEST",
        "crop":       "wheat",
        "area_acres": 30.0,
        "lat":        36.7720,
        "lon":        -119.4300,
    },
]

# Use the first zone's coordinates as the farm centroid for weather queries
_FARM_LAT = ZONES[0]["lat"]
_FARM_LON = ZONES[0]["lon"]


# =============================================================================
# SIMULATION FALLBACK
# Used whenever a real API call fails or its env var is not set.
# Keeps the system running with realistic demo data instead of crashing.
# =============================================================================

def _rng(salt: int = 0) -> random.Random:
    """Date-seeded RNG — identical within one calendar day."""
    seed = int(date.today().strftime("%Y%m%d")) + salt
    return random.Random(seed)


def _sim_weather() -> Dict:
    r = _rng(1)
    temp = round(r.uniform(18, 38), 1)
    return {
        "temperature_c":      temp,
        "humidity_percent":   round(r.uniform(30, 75), 1),
        "precipitation_mm":   round(r.uniform(0, 12), 1),
        "wind_speed_kmh":     round(r.uniform(5, 45), 1),
        "frost_risk":         temp < 2.0,
        "heat_stress_risk":   temp > 35.0,
        "severe_weather_alert": "None",
        "forecast_7day": [
            {
                "day":          i + 1,
                "temp_high_c":  round(temp + _rng(i + 10).uniform(-5, 5), 1),
                "temp_low_c":   round(temp + _rng(i + 20).uniform(-10, 0), 1),
                "precip_mm":    round(_rng(i + 30).uniform(0, 15), 1),
                "condition":    _rng(i + 40).choice(
                                    ["sunny", "partly_cloudy", "overcast", "rain"]),
            }
            for i in range(7)
        ],
    }


def _sim_soil() -> List[Dict]:
    readings = []
    for i, zone in enumerate(ZONES):
        r = _rng(100 + i)
        readings.append({
            "zone_id":            zone["zone_id"],
            "crop":               zone["crop"],
            "moisture_percent":   round(r.uniform(15, 75), 1),
            "temperature_c":      round(r.uniform(18, 32), 1),
            "ph":                 round(r.uniform(5.8, 7.2), 2),
            "nitrogen_ppm":       round(r.uniform(20, 180), 1),
            "phosphorus_ppm":     round(r.uniform(10, 80), 1),
            "potassium_ppm":      round(r.uniform(80, 250), 1),
            "organic_matter_pct": round(r.uniform(1.5, 4.5), 2),
            "compaction_index":   round(r.uniform(10, 70), 1),
        })
    return readings


def _sim_crop() -> List[Dict]:
    stages = ["germination", "seedling", "vegetative",
              "flowering", "grain_fill", "maturation"]
    readings = []
    for i, zone in enumerate(ZONES):
        r = _rng(200 + i)
        readings.append({
            "zone_id":             zone["zone_id"],
            "crop":                zone["crop"],
            "area_acres":          zone["area_acres"],
            "growth_stage":        r.choice(stages),
            "days_since_planting": r.randint(20, 90),
            "days_to_harvest":     r.randint(10, 60),
            "canopy_coverage_pct": round(r.uniform(40, 95), 1),
            "ndvi":                round(r.uniform(0.3, 0.85), 3),
            "pest_pressure":       r.choice(["none", "none", "low", "medium", "high"]),
            "disease_risk":        r.choice(["none", "none", "low", "medium"]),
            "yield_forecast_t_ha": round(r.uniform(3.5, 9.5), 2),
        })
    return readings


def _sim_market() -> Dict:
    r = _rng(300)
    return {
        "wheat_usd_per_bushel":    round(r.uniform(5.5, 8.5), 2),
        "corn_usd_per_bushel":     round(r.uniform(4.2, 6.8), 2),
        "soybeans_usd_per_bushel": round(r.uniform(11.0, 16.5), 2),
        "as_of":                   datetime.now().isoformat(timespec="minutes"),
    }


# =============================================================================
# 1. WEATHER — OpenWeatherMap API
# =============================================================================
# API docs: https://openweathermap.org/current
# Free tier: 1,000 calls/day, hourly current + 5-day/3-hour forecast
# Paid tier: One Call API 3.0 gives hourly + 8-day daily forecast
#
# REPLACE with Tomorrow.io if you need:
#   - Minute-level precipitation nowcasting
#   - Field-level hyper-local forecasts
#   API docs: https://docs.tomorrow.io/reference/realtime-weather
# =============================================================================

def get_weather() -> Dict:
    """
    Fetch current weather and 7-day forecast for the farm location.

    Returns the same dict structure as simulation.py:
    {
        temperature_c, humidity_percent, precipitation_mm,
        wind_speed_kmh, frost_risk, heat_stress_risk,
        severe_weather_alert, forecast_7day: [ {day, temp_high_c,
        temp_low_c, precip_mm, condition}, ... ]
    }
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
    if not api_key:
        log.warning("[weather] OPENWEATHERMAP_API_KEY not set — using simulation data")
        return _sim_weather()

    try:
        import requests

        # ── Current conditions ─────────────────────────────────────────────
        current_url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={_FARM_LAT}&lon={_FARM_LON}"
            f"&units=metric&appid={api_key}"
        )
        current_resp = requests.get(current_url, timeout=10)
        current_resp.raise_for_status()
        cw = current_resp.json()

        temp_c    = cw["main"]["temp"]
        humid_pct = cw["main"]["humidity"]
        wind_kmh  = round(cw["wind"]["speed"] * 3.6, 1)  # m/s to km/h
        precip_mm = cw.get("rain", {}).get("1h", 0.0)
        weather_id = cw["weather"][0]["id"]

        # Severe weather: OWM weather IDs 200-699 cover storms, rain, snow
        severe_alert = "None"
        if weather_id < 300:
            severe_alert = f"Thunderstorm warning — OWM code {weather_id}"
        elif weather_id < 600 and cw["weather"][0].get("main") == "Snow":
            severe_alert = f"Snow warning — OWM code {weather_id}"

        # ── 5-day / 3-hour forecast ────────────────────────────────────────
        # OWM free tier returns 40 x 3-hour slots — aggregate to daily
        forecast_url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={_FARM_LAT}&lon={_FARM_LON}"
            f"&units=metric&appid={api_key}"
        )
        fc_resp = requests.get(forecast_url, timeout=10)
        fc_resp.raise_for_status()
        fc_data = fc_resp.json()

        # Group 3-hour slots by date and derive daily high/low/precip
        daily: Dict[str, dict] = {}
        for slot in fc_data["list"]:
            slot_date = slot["dt_txt"][:10]          # "YYYY-MM-DD"
            if slot_date not in daily:
                daily[slot_date] = {"highs": [], "lows": [], "precip": 0.0,
                                    "conditions": []}
            daily[slot_date]["highs"].append(slot["main"]["temp_max"])
            daily[slot_date]["lows"].append(slot["main"]["temp_min"])
            daily[slot_date]["precip"] += slot.get("rain", {}).get("3h", 0.0)
            daily[slot_date]["conditions"].append(
                slot["weather"][0]["main"].lower())

        _condition_map = {
            "thunderstorm": "rain", "drizzle": "rain", "rain": "rain",
            "snow": "overcast",     "clouds":  "partly_cloudy",
            "clear": "sunny",
        }

        forecast_7day = []
        for day_num, (day_str, vals) in enumerate(
                sorted(daily.items())[:7], start=1):
            most_common_cond = max(
                set(vals["conditions"]),
                key=vals["conditions"].count)
            forecast_7day.append({
                "day":         day_num,
                "temp_high_c": round(max(vals["highs"]), 1),
                "temp_low_c":  round(min(vals["lows"]), 1),
                "precip_mm":   round(vals["precip"], 1),
                "condition":   _condition_map.get(most_common_cond,
                                                  "partly_cloudy"),
            })

        # Pad to 7 days with simulation if OWM only returned 5
        sim_fc = _sim_weather()["forecast_7day"]
        while len(forecast_7day) < 7:
            forecast_7day.append(sim_fc[len(forecast_7day)])

        log.info("[weather] OpenWeatherMap OK — temp=%.1f°C, precip=%.1fmm",
                 temp_c, precip_mm)

        return {
            "temperature_c":        round(temp_c, 1),
            "humidity_percent":     round(humid_pct, 1),
            "precipitation_mm":     round(precip_mm, 1),
            "wind_speed_kmh":       wind_kmh,
            "frost_risk":           temp_c < float(
                                        os.getenv("FROST_WARNING_TEMP_C", "2.0")),
            "heat_stress_risk":     temp_c > float(
                                        os.getenv("HEAT_STRESS_TEMP_C", "35.0")),
            "severe_weather_alert": severe_alert,
            "forecast_7day":        forecast_7day,
        }

    except Exception as exc:
        log.error("[weather] OpenWeatherMap call failed: %s — using simulation", exc)
        return _sim_weather()


# =============================================================================
# 2. SOIL SENSORS — Sentek / Stevens HydraProbe LoRaWAN Gateway
# =============================================================================
# This implementation calls a generic REST gateway that aggregates readings
# from all in-field soil sensors (Sentek EnviroSCAN, Stevens HydraProbe,
# or compatible LoRaWAN sensors) and returns JSON.
#
# GATEWAY RESPONSE FORMAT EXPECTED:
# The gateway endpoint GET /sensors/readings should return:
# [
#   {
#     "zone_id": "Z-NORTH",
#     "moisture_vwc": 0.32,          <- volumetric water content 0.0-1.0
#     "temperature_c": 22.5,
#     "ec_ds_m": 0.45,               <- electrical conductivity (proxy for nutrients)
#     "ph": 6.4,                     <- if pH probe installed, else null
#     "nitrogen_ppm": 85.0,          <- if nutrient probe installed, else null
#     "phosphorus_ppm": 42.0,
#     "potassium_ppm": 180.0,
#     "organic_matter_pct": 2.8,     <- from lab analysis, updated periodically
#     "compaction_index": 32.0       <- from penetrometer sensor or field survey
#   },
#   ...one entry per zone...
# ]
#
# If your gateway returns a different format, edit _parse_gateway_response()
# below to map your field names to the expected structure.
#
# ALTERNATIVE GATEWAY APIS:
#   Teralytic (nutrient + moisture): https://teralytic.com/api/
#   CropX:                           https://cropx.com/api/
#   AquaSpy:                         https://aquaspy.com/
# =============================================================================

def _parse_gateway_response(raw: List[dict], zone: dict) -> Optional[dict]:
    """
    Map raw gateway JSON to the structure expected by the agent.
    Edit this function to match your gateway's field names.
    """
    # Find the entry for this zone
    entry = next((r for r in raw if r.get("zone_id") == zone["zone_id"]), None)
    if entry is None:
        return None

    # Convert volumetric water content (0.0-1.0) to percent if needed
    moisture = entry.get("moisture_vwc", entry.get("moisture_percent", 0))
    if moisture <= 1.0:
        moisture = round(moisture * 100, 1)   # VWC 0.32 -> 32.0%

    return {
        "zone_id":            zone["zone_id"],
        "crop":               zone["crop"],
        "moisture_percent":   moisture,
        "temperature_c":      round(float(entry.get("temperature_c", 20.0)), 1),
        "ph":                 round(float(entry.get("ph") or 6.5), 2),
        "nitrogen_ppm":       round(float(entry.get("nitrogen_ppm") or 80.0), 1),
        "phosphorus_ppm":     round(float(entry.get("phosphorus_ppm") or 40.0), 1),
        "potassium_ppm":      round(float(entry.get("potassium_ppm") or 160.0), 1),
        "organic_matter_pct": round(float(entry.get("organic_matter_pct") or 2.5), 2),
        "compaction_index":   round(float(entry.get("compaction_index") or 30.0), 1),
    }


def get_soil_readings() -> List[Dict]:
    """
    Fetch current soil sensor readings for all farm zones.

    Returns the same list structure as simulation.py — one dict per zone:
    [{ zone_id, crop, moisture_percent, temperature_c, ph,
       nitrogen_ppm, phosphorus_ppm, potassium_ppm,
       organic_matter_pct, compaction_index }, ...]
    """
    api_url = os.getenv("SOIL_SENSOR_API_URL", "")
    api_key = os.getenv("SOIL_SENSOR_API_KEY", "")

    if not api_url:
        log.warning("[soil] SOIL_SENSOR_API_URL not set — using simulation data")
        return _sim_soil()

    try:
        import requests

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        endpoint = f"{api_url.rstrip('/')}/sensors/readings"
        resp = requests.get(endpoint, headers=headers, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        readings = []
        sim_fallback = _sim_soil()

        for i, zone in enumerate(ZONES):
            parsed = _parse_gateway_response(raw, zone)
            if parsed:
                readings.append(parsed)
                log.info("[soil] %s — moisture=%.1f%%, pH=%.2f",
                         zone["zone_id"],
                         parsed["moisture_percent"],
                         parsed["ph"])
            else:
                # Zone not returned by gateway — use simulation for this zone
                log.warning("[soil] %s not found in gateway response — using simulation",
                            zone["zone_id"])
                readings.append(sim_fallback[i])

        return readings

    except Exception as exc:
        log.error("[soil] Gateway call failed: %s — using simulation", exc)
        return _sim_soil()


# =============================================================================
# 3. CROP GROWTH — Google Earth Engine (Sentinel-2 NDVI)
# =============================================================================
# Uses Sentinel-2 Level-2A surface reflectance imagery to compute NDVI
# (Normalised Difference Vegetation Index) per zone.
# NDVI range: 0.0 (bare soil) to 1.0 (dense healthy vegetation)
#
# SETUP:
#   1. Create a Google Earth Engine account: https://earthengine.google.com
#   2. Create a Cloud project and enable the Earth Engine API
#   3. Authenticate: earthengine authenticate
#   4. Set GEE_PROJECT_ID in .env
#
# INSTALL:
#   pip install earthengine-api google-auth
#
# NOTE ON GROWTH STAGE AND PEST DATA:
#   GEE only provides NDVI and canopy coverage from satellite imagery.
#   Growth stage (vegetative, flowering, grain_fill, etc.) must come from
#   one of these sources:
#     a) DSSAT or APSIM crop model running locally/cloud
#     b) Farm management platform API (John Deere Operations Center,
#        Climate FieldView, Trimble Ag Software)
#     c) Manual input via your own database table
#   Pest pressure and disease risk also require scouting reports or a
#   dedicated crop health monitoring service (Scouting app, Farmers Edge).
#   This implementation fetches real NDVI from GEE and falls back to
#   simulation for the fields that require crop models or scouting data.
# =============================================================================

def _compute_ndvi_gee(zone: dict, project_id: str) -> Optional[float]:
    """
    Compute median NDVI for one zone polygon using the latest cloud-free
    Sentinel-2 image within the last 14 days.

    Returns NDVI as a float, or None on error.
    """
    try:
        import ee

        # Initialise with the project (only needed once per session)
        try:
            ee.Initialize(project=project_id)
        except ee.EEException:
            pass   # already initialised

        # Build a small bounding box around the zone centroid
        # In production, replace with the actual field polygon geometry
        lat, lon = zone["lat"], zone["lon"]
        buffer  = 0.005    # ~500 metre buffer around centroid
        region  = ee.Geometry.Rectangle(
            [lon - buffer, lat - buffer, lon + buffer, lat + buffer])

        # Latest Sentinel-2 SR image, cloud cover < 20%, last 14 days
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(
                (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d"),
                datetime.utcnow().strftime("%Y-%m-%d"))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .sort("system:time_start", False)   # most recent first
            .first()
        )

        # NDVI = (NIR - Red) / (NIR + Red)
        # Sentinel-2: B8 = NIR (842nm), B4 = Red (665nm)
        ndvi_image = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

        stats = ndvi_image.reduceRegion(
            reducer  = ee.Reducer.mean(),
            geometry = region,
            scale    = 10,      # Sentinel-2 resolution in metres
            maxPixels= 1e6
        )

        ndvi_value = stats.get("NDVI").getInfo()
        if ndvi_value is None:
            return None

        return round(float(ndvi_value), 3)

    except Exception as exc:
        log.warning("[crop][gee] NDVI fetch failed for %s: %s",
                    zone["zone_id"], exc)
        return None


def get_crop_readings() -> List[Dict]:
    """
    Fetch crop growth data for all farm zones.
    NDVI is fetched from Google Earth Engine where available.
    Growth stage, pest pressure, and disease risk fall back to simulation.

    Returns the same list structure as simulation.py — one dict per zone:
    [{ zone_id, crop, area_acres, growth_stage, days_since_planting,
       days_to_harvest, canopy_coverage_pct, ndvi,
       pest_pressure, disease_risk, yield_forecast_t_ha }, ...]
    """
    project_id = os.getenv("GEE_PROJECT_ID", "")
    sim_data   = _sim_crop()   # baseline for fields not from GEE

    if not project_id:
        log.warning("[crop] GEE_PROJECT_ID not set — using simulation data")
        return sim_data

    readings = []
    for i, zone in enumerate(ZONES):
        sim_zone = sim_data[i]

        # Attempt to get real NDVI from GEE
        real_ndvi = _compute_ndvi_gee(zone, project_id)

        if real_ndvi is not None:
            log.info("[crop] %s — real NDVI=%.3f from Sentinel-2",
                     zone["zone_id"], real_ndvi)
            # Derive canopy coverage from NDVI (linear approximation)
            # NDVI 0.3 ≈ 30% cover, NDVI 0.85 ≈ 95% cover
            canopy_pct = round(min(100.0, max(0.0,
                               (real_ndvi - 0.1) / 0.85 * 100)), 1)

            readings.append({
                "zone_id":             zone["zone_id"],
                "crop":                zone["crop"],
                "area_acres":          zone["area_acres"],
                # Growth stage, pest, disease from simulation until crop model integrated
                "growth_stage":        sim_zone["growth_stage"],
                "days_since_planting": sim_zone["days_since_planting"],
                "days_to_harvest":     sim_zone["days_to_harvest"],
                "canopy_coverage_pct": canopy_pct,
                "ndvi":                real_ndvi,             # REAL from Sentinel-2
                "pest_pressure":       sim_zone["pest_pressure"],
                "disease_risk":        sim_zone["disease_risk"],
                "yield_forecast_t_ha": sim_zone["yield_forecast_t_ha"],
            })
        else:
            log.warning("[crop] %s — GEE NDVI unavailable, using simulation",
                        zone["zone_id"])
            readings.append(sim_zone)

    return readings


# =============================================================================
# 4. MARKET PRICES — CME Group DataMine API
# =============================================================================
# CME DataMine provides delayed and real-time futures prices for agricultural
# commodities including CBOT Wheat (ZW), CBOT Corn (ZC), CBOT Soybeans (ZS).
#
# API docs:  https://datamine.cmegroup.com
# Free tier: 10-minute delayed data available on CME DataMine
# Paid tier: Real-time streaming via CME Direct or FIX protocol
#
# ALTERNATIVE: USDA NASS Quick Stats API (free, weekly cash prices)
#   https://quickstats.nass.usda.gov/api
#   Endpoint: https://quickstats.nass.usda.gov/api/api_GET/
#   Parameters: commodity_desc=WHEAT&statisticcat_desc=PRICE+RECEIVED
#
# NOTE ON UNITS:
#   CME futures prices are quoted in US cents per bushel.
#   This function converts to USD per bushel (divide by 100).
# =============================================================================

def get_market_prices() -> Dict:
    """
    Fetch current commodity futures prices for wheat, corn, and soybeans.

    Returns the same dict structure as simulation.py:
    {
        wheat_usd_per_bushel,
        corn_usd_per_bushel,
        soybeans_usd_per_bushel,
        as_of
    }
    """
    api_key = os.getenv("CME_API_KEY", "")

    if not api_key:
        log.warning("[market] CME_API_KEY not set — using simulation data")
        return _sim_market()

    try:
        import requests

        # CME DataMine REST API — delayed quotes endpoint
        # Symbols: ZW (wheat), ZC (corn), ZS (soybeans)
        # Front month (nearest expiry) contract
        base_url = "https://datamine.cmegroup.com/cme/api/v1"
        headers  = {
            "Authorization": f"Bearer {api_key}",
            "Accept":        "application/json",
        }

        prices = {}
        symbol_map = {
            "ZW": "wheat_usd_per_bushel",
            "ZC": "corn_usd_per_bushel",
            "ZS": "soybeans_usd_per_bushel",
        }

        for symbol, field_name in symbol_map.items():
            try:
                # Get the front-month futures quote
                url  = f"{base_url}/quotes/{symbol}"
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                # CME returns price in cents/bushel — convert to USD/bushel
                # The exact field path depends on the API version you are using.
                # Adjust the path below if your response structure differs.
                last_price_cents = (
                    data.get("quote", {}).get("last") or
                    data.get("lastPrice") or
                    data.get("data", [{}])[0].get("last", 0)
                )
                prices[field_name] = round(float(last_price_cents) / 100, 2)
                log.info("[market] %s = $%.2f/bu", symbol, prices[field_name])

            except Exception as exc:
                log.warning("[market] %s price fetch failed: %s — using simulation",
                            symbol, exc)
                sim_val = _sim_market()
                prices[field_name] = sim_val[field_name]

        prices["as_of"] = datetime.now().isoformat(timespec="minutes")
        return prices

    except Exception as exc:
        log.error("[market] CME API failed: %s — using simulation", exc)
        return _sim_market()


# =============================================================================
# USDA NASS Quick Stats API — free alternative for cash prices
# =============================================================================
# Uncomment and use this function instead of get_market_prices() if you do
# not have a CME DataMine subscription. USDA publishes weekly cash prices
# received by farmers — these are closer to actual selling prices than
# CME futures and are completely free.
#
# def get_market_prices() -> Dict:
#     import requests
#     nass_key = os.getenv("USDA_NASS_API_KEY", "")
#     if not nass_key:
#         log.warning("[market] USDA_NASS_API_KEY not set — using simulation data")
#         return _sim_market()
#
#     base = "https://quickstats.nass.usda.gov/api/api_GET/"
#     params_base = {
#         "key":               nass_key,
#         "source_desc":       "SURVEY",
#         "sector_desc":       "CROPS",
#         "statisticcat_desc": "PRICE RECEIVED",
#         "unit_desc":         "$ / BU",
#         "freq_desc":         "WEEKLY",
#         "format":            "JSON",
#     }
#
#     def _fetch_price(commodity: str) -> float:
#         resp = requests.get(base, params={**params_base,
#                             "commodity_desc": commodity.upper()}, timeout=15)
#         resp.raise_for_status()
#         items = resp.json().get("data", [])
#         # Most recent entry
#         sorted_items = sorted(items, key=lambda x: x.get("week_ending", ""),
#                               reverse=True)
#         return float(sorted_items[0]["Value"].replace(",", "")) if sorted_items else 0.0
#
#     try:
#         return {
#             "wheat_usd_per_bushel":    _fetch_price("wheat"),
#             "corn_usd_per_bushel":     _fetch_price("corn"),
#             "soybeans_usd_per_bushel": _fetch_price("soybeans, beans"),
#             "as_of":                   datetime.now().isoformat(timespec="minutes"),
#         }
#     except Exception as exc:
#         log.error("[market] USDA NASS failed: %s — using simulation", exc)
#         return _sim_market()
