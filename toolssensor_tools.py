# tools/sensor_tools.py
#
# Farm sensor data simulation layer.
# All four functions simulate data feeds that would come from:
#
#   fetch_weather_data()          -> Weather APIs: OpenWeatherMap, DTN/The Weather Company,
#                                    Tomorrow.io, NOAA National Weather Service API,
#                                    On-farm weather stations (Davis Instruments, Campbell Scientific)
#                                    Data: temperature, humidity, rainfall, wind, 7-day forecast
#
#   read_soil_sensors()           -> In-ground IoT soil sensors:
#                                    Sentek EnviroSCAN / Diviner 2000 (moisture),
#                                    Stevens HydraProbe (moisture + temperature + EC),
#                                    YSI ProDSS (pH, dissolved oxygen),
#                                    HACH nutrient probes (nitrogen, phosphorus, potassium),
#                                    Veris Technologies MSP (compaction / cone penetrometer)
#                                    Transmitted via LoRaWAN / cellular to farm management platform
#
#   read_crop_growth_sensors()    -> Remote sensing + in-field sensors:
#                                    Drone multispectral imagery (DJI Agras + MicaSense RedEdge),
#                                    Satellite NDVI time series (Planet Labs, Sentinel-2 via ESA),
#                                    Machine vision canopy cameras (Hortau, CropX),
#                                    Leaf wetness / disease risk sensors (Onset HOBO),
#                                    Growth model APIs (NASA DSSAT, APSIM)
#
#   get_historical_farm_data()    -> Farm management information system (FMIS):
#                                    John Deere Operations Center API,
#                                    Climate FieldView API,
#                                    Trimble Ag Software,
#                                    On-premise agronomic database
#
# SIMULATION DESIGN:
#   - Date-seeded RNG: identical values within the same calendar day (reproducible),
#     different values each new day (mimics sensor variability)
#   - Zone-specific soil profiles: each of the 5 zones has documented characteristics
#     calibrated to its crop type and a realistic baseline moisture/pH/nutrient range
#   - Weather uses a Central Valley, CA seasonal baseline:
#       Summer: 28-38°C, low humidity, drought conditions common
#       Simulation runs as mid-summer by default
#   - Growth stage is derived from planting date simulation per zone
#   - All numeric ranges sourced from published agronomic data for wheat/corn/soybean
#
# TO REPLACE WITH REAL DATA:
#   Swap each function body for the real API call or sensor read listed above.
#   All downstream agents and the LangGraph pipeline require NO changes.

import random
import math
from datetime import datetime, timedelta, date

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        return fn

from config.settings import farm_config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng(extra_seed: int = 0) -> random.Random:
    """
    Date-seeded RNG. extra_seed lets zone/crop calls differ from each other
    while still being fully reproducible on the same calendar day.
    """
    seed = int(date.today().strftime("%Y%m%d")) + extra_seed
    return random.Random(seed)


def _day_of_year() -> int:
    return datetime.now().timetuple().tm_yday


# ---------------------------------------------------------------------------
# Static farm data
# In production this comes from the FMIS (Farm Management Information System)
# and GIS field boundary database.
# ---------------------------------------------------------------------------

# Five monitored zones: area, crop, soil type, current growth stage
# Growth stages are derived from planting date + days-since-planting (simulated below)
ZONES = {
    "zone_a": {
        "crop":            "wheat",
        "area_acres":      35,
        "soil_type":       "sandy_loam",
        "field_name":      "North Field A",
        # Soil baseline ranges for this zone (from last soil health report)
        "moisture_range":  (22, 65),    # % volumetric water content
        "ph_range":        (6.2, 7.1),
        "nitrogen_range":  (18, 62),    # ppm
        "phosphorus_range":(8,  32),    # ppm
        "potassium_range": (95, 210),   # ppm
        "organic_matter":  (2.2, 4.1),  # %
        "compaction_range":(15, 55),    # PSI index
        # Planting calendar
        "planted_doy":     75,          # day-of-year planted (mid March for winter wheat)
        "typical_harvest_doy": 195,     # mid July
    },
    "zone_b": {
        "crop":            "corn",
        "area_acres":      40,
        "soil_type":       "clay_loam",
        "field_name":      "East Field B",
        "moisture_range":  (30, 72),
        "ph_range":        (5.9, 6.8),
        "nitrogen_range":  (25, 78),
        "phosphorus_range":(10, 38),
        "potassium_range": (110, 240),
        "organic_matter":  (2.8, 5.2),
        "compaction_range":(20, 65),
        "planted_doy":     120,         # late April for corn
        "typical_harvest_doy": 280,     # early October
    },
    "zone_c": {
        "crop":            "soybeans",
        "area_acres":      30,
        "soil_type":       "silt_loam",
        "field_name":      "South Field C",
        "moisture_range":  (25, 68),
        "ph_range":        (6.0, 7.0),
        "nitrogen_range":  (12, 45),    # lower: soybeans fix own N
        "phosphorus_range":(7,  28),
        "potassium_range": (88, 195),
        "organic_matter":  (2.5, 4.5),
        "compaction_range":(12, 50),
        "planted_doy":     130,         # early May for soybeans
        "typical_harvest_doy": 275,     # late September
    },
    "zone_d": {
        "crop":            "wheat",
        "area_acres":      25,
        "soil_type":       "loam",
        "field_name":      "West Field D",
        "moisture_range":  (20, 62),
        "ph_range":        (6.3, 7.2),
        "nitrogen_range":  (15, 58),
        "phosphorus_range":(6,  30),
        "potassium_range": (90, 205),
        "organic_matter":  (2.0, 3.8),
        "compaction_range":(18, 60),
        "planted_doy":     78,
        "typical_harvest_doy": 198,
    },
    "zone_e": {
        "crop":            "corn",
        "area_acres":      20,
        "soil_type":       "sandy_clay_loam",
        "field_name":      "Central Field E",
        "moisture_range":  (28, 70),
        "ph_range":        (6.0, 6.9),
        "nitrogen_range":  (22, 72),
        "phosphorus_range":(9,  35),
        "potassium_range": (105, 230),
        "organic_matter":  (2.4, 4.8),
        "compaction_range":(22, 68),
        "planted_doy":     118,
        "typical_harvest_doy": 278,
    },
}

# Crop growth stages in order with minimum days-since-planting thresholds
# Source: USDA NASS Crop Progress methodology + published crop calendars
_GROWTH_STAGES = {
    "wheat": [
        ("pre_planting",  0),
        ("germination",   7),
        ("seedling",      21),
        ("vegetative",    45),
        ("flowering",     90),
        ("grain_fill",    110),
        ("maturation",    130),
        ("harvest_ready", 145),
    ],
    "corn": [
        ("pre_planting",  0),
        ("germination",   10),
        ("seedling",      25),
        ("vegetative",    55),
        ("flowering",     75),
        ("grain_fill",    95),
        ("maturation",    120),
        ("harvest_ready", 140),
    ],
    "soybeans": [
        ("pre_planting",  0),
        ("germination",   12),
        ("seedling",      28),
        ("vegetative",    50),
        ("flowering",     70),
        ("grain_fill",    90),
        ("maturation",    115),
        ("harvest_ready", 130),
    ],
}

def _growth_stage(crop: str, days_since_planting: int) -> str:
    """Derive growth stage from days since planting using crop calendar."""
    stages = _GROWTH_STAGES.get(crop, _GROWTH_STAGES["wheat"])
    current_stage = stages[0][0]
    for stage_name, min_days in stages:
        if days_since_planting >= min_days:
            current_stage = stage_name
    return current_stage

# Central Valley, CA seasonal weather baseline (simulation target: mid-summer)
# Source: NOAA Climate Normals for Fresno, CA
_WEATHER_BASELINE = {
    "temp_mean_c":   31.0,
    "temp_std_c":     4.5,
    "humidity_mean": 38.0,
    "humidity_std":   9.0,
    "wind_kmh_mean": 16.0,
    "wind_kmh_std":   6.0,
    "precip_prob":    0.06,       # 6% chance of rain in summer
    "precip_mean_mm": 2.5,        # if it does rain, low volume
    "frost_prob":     0.01,       # rare in summer
    "heat_stress_threshold_c": farm_config.heat_stress_temp_c,
}

_WEATHER_CONDITIONS_BY_TEMP = [
    (15, "overcast"),
    (22, "partly_cloudy"),
    (28, "sunny"),
    (33, "sunny"),
    (38, "sunny"),              # hot and dry in Central Valley summer
]

def _temp_to_condition(temp_c: float) -> str:
    for threshold, condition in _WEATHER_CONDITIONS_BY_TEMP:
        if temp_c <= threshold:
            return condition
    return "sunny"


# ---------------------------------------------------------------------------
# Tool 1: Weather data
# ---------------------------------------------------------------------------

@tool
def fetch_weather_data(location: str) -> dict:
    """
    Fetch current weather and 7-day forecast for the farm location.

    Production source:
      - DTN/The Weather Company Agriculture API (hyper-local, 1km grid)
      - On-farm Davis Instruments Vantage Pro2 weather station (real-time)
      - NOAA NWS API (backup + climate normals)
    Data latency in production: 5 minutes (station), 1 hour (API forecast).

    Simulation method:
      - Temperature drawn from N(31°C, 4.5°C) — Central Valley CA summer baseline
      - Precipitation: 6% chance of rain; if raining, amount from Gamma distribution
      - 7-day forecast degrades in certainty: days 1-2 tight range, days 5-7 wider
      - Heat stress if temp > 35°C, frost if temp < 2°C
      - 5% chance of severe weather alert

    Args:
        location: Farm location string (e.g. 'Central Valley, CA')
    Returns:
        Dict with current conditions and 7-day forecast
    """
    rng = _rng(extra_seed=1)
    b   = _WEATHER_BASELINE

    # Current conditions
    temp_c      = round(rng.gauss(b["temp_mean_c"], b["temp_std_c"]), 1)
    humidity    = round(max(10, min(95, rng.gauss(b["humidity_mean"], b["humidity_std"]))), 1)
    wind_kmh    = round(max(0, rng.gauss(b["wind_kmh_mean"], b["wind_kmh_std"])), 1)
    is_raining  = rng.random() < b["precip_prob"]
    precip_mm   = round(rng.uniform(0.5, 8.0), 1) if is_raining else 0.0
    frost_risk  = temp_c < farm_config.frost_warning_temp_c
    heat_stress = temp_c > farm_config.heat_stress_temp_c

    # Severe weather (5% probability)
    severe_alerts = [
        "Hail warning in effect until 6 PM — cover vulnerable crops",
        "Flash flood watch — low-lying fields at risk of waterlogging",
        "High wind advisory — 65+ km/h gusts expected, delay spraying operations",
        "Excessive heat warning — temperatures forecast above 42°C for 3 days",
    ]
    severe_alert = rng.choice(severe_alerts) if rng.random() < 0.05 else None

    # 7-day forecast (uncertainty grows with days ahead)
    forecast = []
    for day_offset in range(1, 8):
        day_date      = date.today() + timedelta(days=day_offset)
        uncertainty   = day_offset * 0.8       # progressively wider range
        f_temp_high   = round(temp_c + rng.gauss(1.5, uncertainty), 1)
        f_temp_low    = round(temp_c + rng.gauss(-7.0, uncertainty), 1)
        f_humidity    = round(max(10, min(95, rng.gauss(humidity, 5.0))), 1)
        f_wind        = round(max(0, rng.gauss(wind_kmh, 4.0)), 1)
        f_precip_prob = b["precip_prob"] + rng.uniform(-0.03, 0.05)
        f_precip      = round(rng.uniform(0.5, 12.0), 1) if rng.random() < f_precip_prob else 0.0
        f_condition   = "light_rain" if f_precip > 0 else _temp_to_condition(f_temp_high)

        forecast.append({
            "date":                day_date.isoformat(),
            "temperature_high_c":  f_temp_high,
            "temperature_low_c":   f_temp_low,
            "humidity_percent":    f_humidity,
            "wind_speed_kmh":      f_wind,
            "precipitation_mm":    f_precip,
            "precipitation_probability": round(f_precip_prob, 2),
            "conditions":          f_condition,
            "field_work_suitable": f_precip == 0 and f_wind < 25,
        })

    return {
        "location":             location,
        "timestamp":            datetime.now().isoformat(),
        "temperature_c":        temp_c,
        "humidity_percent":     humidity,
        "precipitation_mm":     precip_mm,
        "wind_speed_kmh":       wind_kmh,
        "conditions":           _temp_to_condition(temp_c),
        "frost_risk":           frost_risk,
        "heat_stress_risk":     heat_stress,
        "severe_weather_alert": severe_alert,
        "forecast_7day":        forecast,
        # Derived agronomic flags
        "field_operations_today": precip_mm == 0 and wind_kmh < 25,
        "irrigation_demand":      "high" if temp_c > 30 and precip_mm == 0 else
                                  "moderate" if temp_c > 24 else "low",
        "spray_window_open":      wind_kmh < 16 and not is_raining,
    }


# ---------------------------------------------------------------------------
# Tool 2: Soil sensors
# ---------------------------------------------------------------------------

@tool
def read_soil_sensors(zone_ids: list) -> list:
    """
    Read current soil condition from in-ground IoT sensors in each zone.

    Production source:
      - Sentek EnviroSCAN capacitance probes (soil moisture at 4 depths)
      - Stevens HydraProbe II (moisture, temperature, electrical conductivity)
      - Continuous nutrient probes (HACH, YSI) or periodic lab analysis
      - Veris MSP mobile sensor platform (compaction, pulled across field)
    Data latency in production: every 15-30 minutes via LoRaWAN gateway.

    Simulation method:
      - Each zone has documented baseline ranges from its soil health profile
      - Moisture varies within zone's range; near-threshold values trigger agent alerts
      - pH drawn from zone-specific range (soybean zones slightly more acidic)
      - Nutrient levels drawn from zone baseline; lower end indicates depletion
      - Compaction increases with clay-heavy soil types (zone_b highest)

    Args:
        zone_ids: e.g. ['zone_a', 'zone_b', 'zone_c']
    Returns:
        List of soil sensor reading dicts, one per zone
    """
    readings = []
    for i, zone_id in enumerate(zone_ids):
        z   = ZONES.get(zone_id)
        if not z:
            continue
        rng = _rng(extra_seed=i + 100)

        moisture    = round(rng.uniform(*z["moisture_range"]), 1)
        ph          = round(rng.uniform(*z["ph_range"]), 2)
        nitrogen    = round(rng.uniform(*z["nitrogen_range"]), 1)
        phosphorus  = round(rng.uniform(*z["phosphorus_range"]), 1)
        potassium   = round(rng.uniform(*z["potassium_range"]), 1)
        org_matter  = round(rng.uniform(*z["organic_matter"]), 2)
        compaction  = round(rng.uniform(*z["compaction_range"]), 1)
        soil_temp   = round(rng.gauss(24.0, 3.5), 1)   # Central Valley soil temp in summer

        # Derived agronomic flags
        moisture_status = (
            "critical_low"  if moisture < farm_config.critical_soil_moisture_low  else
            "critical_high" if moisture > farm_config.critical_soil_moisture_high else
            "low"           if moisture < 30 else
            "optimal"       if moisture < 65 else
            "high"
        )
        nitrogen_status = (
            "deficient" if nitrogen < 20 else
            "low"       if nitrogen < 35 else
            "optimal"   if nitrogen < 65 else
            "excess"
        )

        readings.append({
            "zone_id":                 zone_id,
            "field_name":              z["field_name"],
            "crop_type":               z["crop"],
            "soil_type":               z["soil_type"],
            "area_acres":              z["area_acres"],
            "timestamp":               datetime.now().isoformat(),
            # Sensor readings
            "moisture_percent":        moisture,
            "temperature_c":           soil_temp,
            "ph_level":                ph,
            "nitrogen_ppm":            nitrogen,
            "phosphorus_ppm":          phosphorus,
            "potassium_ppm":           potassium,
            "organic_matter_percent":  org_matter,
            "compaction_index":        compaction,
            # Derived status flags for quick agent reference
            "moisture_status":         moisture_status,
            "nitrogen_status":         nitrogen_status,
            "irrigation_needed":       moisture < 30,
            "ph_out_of_range":         ph < 5.8 or ph > 7.5,
        })

    return readings


# ---------------------------------------------------------------------------
# Tool 3: Crop growth sensors
# ---------------------------------------------------------------------------

@tool
def read_crop_growth_sensors(zone_ids: list) -> list:
    """
    Read crop growth stage and health data from remote sensing and field cameras.

    Production source:
      - Weekly drone flights (DJI Agras T40 + MicaSense RedEdge-MX):
          NDVI, NDRE, canopy cover, leaf area index
      - Sentinel-2 satellite NDVI (10m resolution, every 5 days via ESA Copernicus)
      - Machine vision canopy cameras (Hortau, CropX) — continuous
      - Disease/pest risk from weather-based models (e.g., BLITECAST for blight)
      - Yield forecast from NASA DSSAT crop simulation model
    Data latency in production: drone = weekly, satellite = 5 days, cameras = hourly.

    Simulation method:
      - Days since planting computed from zone's planted_doy and today's day-of-year
      - Growth stage derived deterministically from days-since-planting + crop calendar
      - NDVI proxy: canopy cover increases with growth stage (0.3 seedling → 0.9 grain_fill)
      - Pest/disease risk weighted: none=50%, low=30%, medium=15%, high=5%
      - Yield forecast available from flowering stage onward

    Args:
        zone_ids: e.g. ['zone_a', 'zone_c']
    Returns:
        List of crop growth status dicts, one per zone
    """
    readings = []
    doy      = _day_of_year()

    for i, zone_id in enumerate(zone_ids):
        z   = ZONES.get(zone_id)
        if not z:
            continue
        rng = _rng(extra_seed=i + 200)

        crop              = z["crop"]
        planted_doy       = z["planted_doy"]
        harvest_doy       = z["typical_harvest_doy"]
        days_since_plant  = max(0, doy - planted_doy)
        days_to_harvest   = max(0, harvest_doy - doy)
        stage             = _growth_stage(crop, days_since_plant)

        # Stage index for canopy/NDVI calculations
        stage_names = [s[0] for s in _GROWTH_STAGES[crop]]
        stage_idx   = stage_names.index(stage) if stage in stage_names else 0
        stage_frac  = stage_idx / (len(stage_names) - 1)   # 0.0–1.0

        # Canopy coverage grows with stage: 10% seedling → 85% grain fill → 70% maturation
        canopy_base   = 10 + (85 * stage_frac if stage_frac < 0.7 else 85 - (stage_frac - 0.7)*50)
        canopy        = round(min(95, max(5, canopy_base + rng.gauss(0, 5))), 1)

        # Leaf Area Index (LAI): 0.3 early → 5.2 peak
        lai_base      = 0.3 + stage_frac * 4.9
        lai           = round(max(0.1, lai_base + rng.gauss(0, 0.3)), 2)

        # Pest and disease risk
        pest_weights    = [50, 30, 15, 5]
        disease_weights = [55, 25, 15, 5]
        pest_risk    = rng.choices(["none","low","medium","high"], weights=pest_weights)[0]
        disease_risk = rng.choices(["none","low","medium","high"], weights=disease_weights)[0]

        # Yield forecast available from flowering onward
        yield_forecast = None
        if stage in ["flowering","grain_fill","maturation","harvest_ready"]:
            # kg/ha from published averages: wheat=4500, corn=10000, soy=3200
            yield_bases = {"wheat":4500,"corn":10000,"soybeans":3200}
            yield_base  = yield_bases.get(crop, 4000)
            yield_forecast = round(yield_base * rng.uniform(0.80, 1.20), 0)

        # Chlorophyll / vigor index (proxy for NDRE): declines at maturation
        vigor_index = round(0.85 * stage_frac * (1 - max(0, stage_frac - 0.75)) + rng.gauss(0, 0.04), 2)

        readings.append({
            "zone_id":                     zone_id,
            "field_name":                  z["field_name"],
            "crop_type":                   crop,
            "area_acres":                  z["area_acres"],
            "timestamp":                   datetime.now().isoformat(),
            # Growth timeline
            "growth_stage":                stage,
            "days_since_planting":         days_since_plant,
            "estimated_days_to_harvest":   days_to_harvest,
            "planted_day_of_year":         planted_doy,
            # Remote sensing indices
            "canopy_coverage_percent":     canopy,
            "leaf_area_index":             lai,
            "vigor_index":                 vigor_index,     # NDRE proxy, 0–1
            # Risk assessments
            "pest_pressure_level":         pest_risk,
            "disease_risk":                disease_risk,
            "stress_level": (
                "high"     if pest_risk == "high" or disease_risk == "high" else
                "moderate" if pest_risk in ["medium"] or disease_risk in ["medium"] else
                "low"
            ),
            # Yield
            "yield_forecast_kg_per_ha":    yield_forecast,
            # Readiness flags
            "harvest_imminent":            stage == "harvest_ready" and days_to_harvest <= 7,
            "intervention_recommended":    pest_risk in ["medium","high"] or disease_risk in ["medium","high"],
        })

    return readings


# ---------------------------------------------------------------------------
# Tool 4: Historical farm data
# ---------------------------------------------------------------------------

@tool
def get_historical_farm_data(zone_id: str, days_back: int = 30) -> dict:
    """
    Retrieve historical sensor logs and agronomic records for a farm zone.

    Production source:
      - John Deere Operations Center API (machine telematics + field records)
      - Climate FieldView API (soil sampling, yield maps, application records)
      - On-premise agronomic database (irrigation logs, spray records, labor)
    Data latency in production: T+1 day for machine data, real-time for manual logs.

    Simulation method:
      - Soil moisture averages drawn from zone's documented baseline
      - Irrigation events: 2-10 per 30 days depending on crop water demand
      - Total irrigation volume calibrated to crop ET (evapotranspiration) needs
      - Previous season yield from published county averages ± 15%
      - pH and nutrient trends from zone baseline

    Args:
        zone_id: Farm zone e.g. 'zone_b'
        days_back: Days of history to summarize
    Returns:
        Dict with aggregated historical agronomic statistics
    """
    z   = ZONES.get(zone_id, ZONES["zone_a"])
    rng = _rng(extra_seed=hash(zone_id) % 1000 + 300)
    crop = z["crop"]

    # Crop water demand (mm/day ET reference): corn > soybeans > wheat in summer
    et_rates = {"corn": 7.5, "soybeans": 6.2, "wheat": 4.8}
    et_daily  = et_rates.get(crop, 6.0)

    # Irrigation: if ET = 7.5 mm/day and no rain, need ~225 mm over 30 days
    # 1 mm per hectare = 10,000 liters; zone is z["area_acres"] × 0.405 ha
    zone_ha           = z["area_acres"] * 0.405
    irr_events        = rng.randint(3, 10)
    total_irr_mm      = round(et_daily * days_back * rng.uniform(0.55, 0.85), 1)  # not 100% replacement
    total_irr_liters  = round(total_irr_mm * 10 * zone_ha, 0)

    yield_bases   = {"wheat":4500,"corn":10000,"soybeans":3200}
    prev_yield    = round(yield_bases.get(crop, 4000) * rng.uniform(0.85, 1.15), 0)

    avg_moisture  = round(rng.uniform(*z["moisture_range"]), 1)
    avg_temp_c    = round(rng.gauss(24.0, 2.5), 1)
    total_precip  = round(rng.uniform(5, 45), 1)  # limited rain in Central Valley summer

    return {
        "zone_id":                          zone_id,
        "field_name":                       z["field_name"],
        "crop_type":                        crop,
        "period_days":                      days_back,

        # Soil health history
        "avg_soil_moisture_percent":        avg_moisture,
        "avg_soil_temperature_c":           avg_temp_c,
        "total_precipitation_mm":           total_precip,
        "soil_ph_trend":                    rng.choice(["stable","acidifying","stable","alkalizing"]),
        "nutrient_depletion_risk":          rng.choice(["low","moderate","moderate","high"]),

        # Irrigation history
        "irrigation_events":                irr_events,
        "total_irrigation_liters":          total_irr_liters,
        "avg_irrigation_volume_per_event_liters": round(total_irr_liters / max(irr_events, 1), 0),
        "estimated_et_mm":                  round(et_daily * days_back, 1),
        "irrigation_efficiency_percent":    round(rng.uniform(72, 90), 1),

        # Yield history
        "avg_yield_previous_season_kg_per_ha": prev_yield,
        "best_season_yield_kg_per_ha":         round(prev_yield * rng.uniform(1.05, 1.25), 0),
        "worst_season_yield_kg_per_ha":        round(prev_yield * rng.uniform(0.70, 0.90), 0),

        # Pest/disease history
        "pest_events_count":                rng.randint(0, 4),
        "fungicide_applications":           rng.randint(0, 2),
        "herbicide_applications":           rng.randint(1, 3),

        # Equipment / labor
        "field_operations_count":           rng.randint(4, 12),
        "equipment_hours":                  round(rng.uniform(8, 45), 1),
    }

