# data/simulation.py
# Deterministic simulation for Farm Management AI Agent v2.
# Date-seeded RNG — same values all day, different values each new day.
#
# Production replacements:
#   fetch_weather()      -> OpenWeatherMap API / Tomorrow.io / NOAA NWS
#   read_soil()          -> Sentek / Stevens HydraProbe LoRaWAN sensors
#   read_crop_growth()   -> Sentinel-2 NDVI / DJI drone multispectral / DSSAT model
#   get_market_prices()  -> CME Group API / USDA NASS Quick Stats API

import random
from datetime import datetime, date, timedelta
from typing import List, Dict

ZONES = [
    {"zone_id": "Z-NORTH",  "crop": "wheat",    "area_acres": 40.0},
    {"zone_id": "Z-SOUTH",  "crop": "corn",     "area_acres": 45.0},
    {"zone_id": "Z-EAST",   "crop": "soybeans", "area_acres": 35.0},
    {"zone_id": "Z-WEST",   "crop": "wheat",    "area_acres": 30.0},
]


def _rng(salt: int = 0) -> random.Random:
    seed = int(date.today().strftime("%Y%m%d")) + salt
    return random.Random(seed)


def get_weather() -> Dict:
    r = _rng(1)
    temp = round(r.uniform(18, 38), 1)
    precip = round(r.uniform(0, 12), 1)
    humid = round(r.uniform(30, 75), 1)
    wind = round(r.uniform(5, 45), 1)
    return {
        "temperature_c":      temp,
        "humidity_percent":   humid,
        "precipitation_mm":   precip,
        "wind_speed_kmh":     wind,
        "frost_risk":         temp < 2.0,
        "heat_stress_risk":   temp > 35.0,
        "severe_weather_alert": "None",
        "forecast_7day": [
            {
                "day": i + 1,
                "temp_high_c": round(temp + _rng(i + 10).uniform(-5, 5), 1),
                "temp_low_c":  round(temp + _rng(i + 20).uniform(-10, 0), 1),
                "precip_mm":   round(_rng(i + 30).uniform(0, 15), 1),
                "condition":   _rng(i + 40).choice(["sunny", "partly_cloudy", "overcast", "rain"]),
            }
            for i in range(7)
        ],
    }


def get_soil_readings() -> List[Dict]:
    readings = []
    for i, zone in enumerate(ZONES):
        r = _rng(100 + i)
        readings.append({
            "zone_id":               zone["zone_id"],
            "crop":                  zone["crop"],
            "moisture_percent":      round(r.uniform(15, 75), 1),
            "temperature_c":         round(r.uniform(18, 32), 1),
            "ph":                    round(r.uniform(5.8, 7.2), 2),
            "nitrogen_ppm":          round(r.uniform(20, 180), 1),
            "phosphorus_ppm":        round(r.uniform(10, 80), 1),
            "potassium_ppm":         round(r.uniform(80, 250), 1),
            "organic_matter_pct":    round(r.uniform(1.5, 4.5), 2),
            "compaction_index":      round(r.uniform(10, 70), 1),
        })
    return readings


def get_crop_readings() -> List[Dict]:
    stages = ["germination", "seedling", "vegetative", "flowering", "grain_fill", "maturation"]
    readings = []
    for i, zone in enumerate(ZONES):
        r = _rng(200 + i)
        readings.append({
            "zone_id":              zone["zone_id"],
            "crop":                 zone["crop"],
            "area_acres":           zone["area_acres"],
            "growth_stage":         r.choice(stages),
            "days_since_planting":  r.randint(20, 90),
            "days_to_harvest":      r.randint(10, 60),
            "canopy_coverage_pct":  round(r.uniform(40, 95), 1),
            "ndvi":                 round(r.uniform(0.3, 0.85), 3),
            "pest_pressure":        r.choice(["none", "none", "low", "medium", "high"]),
            "disease_risk":         r.choice(["none", "none", "low", "medium"]),
            "yield_forecast_t_ha":  round(r.uniform(3.5, 9.5), 2),
        })
    return readings


def get_market_prices() -> Dict:
    r = _rng(300)
    return {
        "wheat_usd_per_bushel":    round(r.uniform(5.5, 8.5), 2),
        "corn_usd_per_bushel":     round(r.uniform(4.2, 6.8), 2),
        "soybeans_usd_per_bushel": round(r.uniform(11.0, 16.5), 2),
        "as_of":                   datetime.now().isoformat(timespec="minutes"),
    }
