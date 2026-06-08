# 🚦 Traffic Demand Prediction
### ML Competition — Ensemble Solution (LightGBM + XGBoost + CatBoost)

<p align="center">
  <img src="https://img.shields.io/badge/Score-91%2B%2F100-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-3%20Ensemble-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-yellow?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>

---

## 🏆 Results

| Model | OOF R² Score |
|-------|-------------|
| 🟢 LightGBM | ~90+ / 100 |
| 🟠 XGBoost | ~90+ / 100 |
| 🔵 CatBoost | ~91+ / 100 |
| 🏆 **Ensemble (Final)** | **91–95 / 100** |

---

## 📌 Overview

This project predicts **traffic demand** across geospatial locations and time slots using a powerful 3-model ensemble. The core insight: **Day 48 historical data serves as an almost perfect lookup table for Day 49 predictions**, combined with rich time, weather, and road features.

```
Day 48 (Training)  ──▶  All 96 timestamps (00:00 → 23:45)
Day 49 (Test)      ──▶  Timestamps 02:15 → 13:45 (47 slots)
                         88.9% have exact Day 48 match ✅
```

---

## 🧠 Key Approach

### 1. Day 48 Lookup Tables (Most Powerful Feature)
Day 48 demand at the same `geohash + timestamp` is used as the primary signal for Day 49 predictions. Multiple resolution levels are built:

```
geohash + exact timestamp  →  finest grain (88.9% test coverage)
geohash + hour             →  hourly fallback
geohash + time block       →  15-min block fallback  
geohash alone              →  location baseline
geo4 / geo3 prefix         →  district-level fallback
```

### 2. Rich Feature Engineering
- **Spatial**: Geohash decoded to lat/lon, distance from center, geo hierarchy (prefix 3/4/5)
- **Temporal**: Cyclic encoding (sin/cos), rush hour flags, part-of-day buckets, day-of-week
- **Weather**: Temperature interactions, hot/cold flags, severe weather indicator
- **Road**: Lane capacity, landmark proximity, road type, peak × lanes interaction
- **Target Encoding**: Out-of-fold encoding for geohash, geo4, geo3, road type, weather

### 3. Three-Model Ensemble with Optimal Weights
Weights are optimized via Nelder-Mead on OOF predictions — no guessing.

---

## 📁 Project Structure

```
traffic-demand-prediction/
│
├── solution.py            # Full pipeline (single file)
├── train.csv              # Training data (not included in repo)
├── test.csv               # Test data (not included in repo)
├── sample_submission.csv  # Submission format
├── submission.csv         # Generated output
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/traffic-demand-prediction.git
cd traffic-demand-prediction

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
lightgbm
xgboost
catboost
scikit-learn
scipy
pygeohash
```

---

## 🚀 Usage

1. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the project root.
2. Run the solution:

```bash
python solution.py
```

3. Output: `submission.csv` with 41,778 demand predictions.

---

## 🔬 Feature Importance (Top Signals)

```
1. d48_ts_mean       ← Day 48 demand at exact location + timestamp
2. d48_hr_mean       ← Day 48 demand at location + hour  
3. d48_geo_mean      ← Overall location demand baseline
4. d48_blk_mean      ← 15-min block level demand
5. geo_ts_k_te       ← Target encoding: geohash × timestamp
6. agg_geo_blk_mean  ← Aggregated block-level stats
7. d48_geo4_hr_mean  ← District × hour demand
8. temp_x_hr         ← Temperature × hour interaction
9. ratio_ts_vs_geo   ← How this slot compares to location average
10. lanes_peak        ← Road capacity during peak hours
```

---

## 📊 Data Schema

| Column | Description |
|--------|-------------|
| `geohash` | Encoded geographic location |
| `day` | Day number (48 = train, 49 = test) |
| `timestamp` | Time in `H:MM` format (15-min intervals) |
| `demand` | Normalized traffic demand (0–1) ← **target** |
| `RoadType` | Residential / Street / Highway |
| `NumberofLanes` | Lane count |
| `LargeVehicles` | Allowed / Not Allowed |
| `Landmarks` | Yes / No |
| `Temperature` | Degrees Celsius |
| `Weather` | Sunny / Rainy / Foggy / Snowy |

---

## 🏗️ Pipeline Summary

```python
# Step 1 — Parse timestamps (H:M format)
# Step 2 — Build Day48 lookup tables at 5 resolution levels
# Step 3 — Engineer 60+ features (spatial, temporal, weather, road)
# Step 4 — Out-of-fold target encoding (5 folds, no leakage)
# Step 5 — Train LightGBM  (5-fold CV, early stopping)
# Step 6 — Train XGBoost   (5-fold CV, early stopping)
# Step 7 — Train CatBoost  (5-fold CV, early stopping)
# Step 8 — Optimize blend weights via Nelder-Mead
# Step 9 — Generate submission.csv
```

---

## 📈 What Made the Difference

| Technique | Impact |
|-----------|--------|
| Day 48 → Day 49 exact lookup | 🔥 Highest single feature importance |
| Multi-resolution geo fallback | Handles unseen geohashes gracefully |
| Leakage-free target encoding | Reliable OOF scores |
| 3-model ensemble | +1–2% over single best model |
| Nelder-Mead weight optimization | Better than fixed 1/3–1/3–1/3 blend |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

<p align="center">Made with ❤️ and gradient boosting</p>