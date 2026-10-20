# 🏏 IPL Moneyball AI: Mathematical Team Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PuLP](https://img.shields.io/badge/Linear%20Programming-PuLP-orange)
![Tests](https://img.shields.io/badge/Tests-24%20passing-brightgreen)

> *"It's about getting things down to one number. Using the stats the way we read them, we'll find value in players that nobody else can see."* — Moneyball (2011)

A data-driven IPL team optimizer that applies **Linear Programming** and **Value Analytics** to identify undervalued players and construct the highest-performing squad within a given auction budget — inspired directly by the Moneyball philosophy.

---

## 🎯 The Core Problem

Traditional IPL auction strategy is driven by reputation, hype, and gut feel. This leads to systematically overpaying for big names while overlooking statistically superior but less famous players.

**This tool solves it in two stages:**

1. **Market Analysis** — Score every player, plot the price-vs-performance landscape, and surface the *efficient frontier*: the set of players who offer the best possible performance at each price point. Players below this frontier are overpriced; those above are hidden gems.

2. **LP Optimization** — Formulate team selection as a **constrained Integer Linear Program** using `PuLP`. The solver guarantees a *mathematically optimal* 11-player squad that maximizes total fantasy score while strictly respecting IPL auction rules.

---

## 🧠 Methodology

### Step 1: Fantasy Score (Composite Performance Metric)

Each player's performance across batting and bowling is condensed into a single `Fantasy_Score` using **Min-Max normalization**:

```math
Fantasy_Score =
  (norm_Runs × 1.5 + norm_SR × 0.75 + norm_Avg × 0.75) × w_bat
+ (norm_Wkts × 1.5 + norm_Econ × 0.75 + norm_BowlAvg × 0.75) × w_bowl
```

- All metrics are min-max scaled to `[0, 1]` using only **qualified** players (≥100 runs or ≥5 wickets) as boundaries — preventing tail-enders with inflated small-sample stats from gaming the score.
- Players with fewer than 30 runs or 2 wickets have their secondary metrics penalized by 80% to further guard against sample noise.
- Strategy weights: `w_bat = w_bowl = 1.0` (Balanced), `1.2 / 0.8` (Batting Heavy), `0.8 / 1.2` (Bowling Heavy).

### Step 2: Value Score (The Moneyball Metric)

```math
Value_Score = Fantasy_Score / Auction_Price_Cr
```

This single number answers *"how much output do you get per crore spent?"* — the exact question Moneyball asks about baseball players. Players with a high Value Score but low absolute cost are the auction targets.

### Step 3: Efficient Frontier

For each player, we check whether any other player exists with **both a higher Fantasy Score and a lower cost**. If so, the player is *dominated* — they offer worse value on both dimensions. The remaining players form the **Pareto-optimal efficient frontier**: the rational picks at each price point.

### Step 4: Integer Linear Programming

Team selection is formulated as a **Binary ILP** solved by PuLP + CBC:

| Element | Definition |
|---|---|
| Decision variable | `x_i ∈ {0, 1}` — 1 if player i is selected |
| Objective | Maximise `Σ x_i × Fantasy_Score_i` |
| Squad size | `Σ x_i = 11` |
| Budget cap | `Σ x_i × Cost_i ≤ Budget` |
| Overseas limit | `Σ x_i × is_foreign_i ≤ 4` |
| Composition | At least 1 WK, ≥ 5 bowling options, ≥ 6 batting options |
| Optional | Forced inclusions / exclusions, exact role counts (Strict mode) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Core Language | Python 3.9+ |
| Data Manipulation | Pandas, NumPy |
| Optimization Engine | PuLP (CBC / GLPK solver) |
| Web Dashboard | Streamlit |
| Visualizations | Plotly Express, Plotly Graph Objects |
| Testing | pytest — 23 unit tests |

---

## 📂 Repository Structure

```text
IPL-Moneyball/
├── data/
│   └── raw/
│       └── IPL_Stats.csv         # 2024–2025 IPL player performance stats
├── src/
│   ├── data_cleaning.py          # Data loading, deduplication, type enforcement
│   └── optimizer.py              # Fantasy scoring, LP model, efficient frontier
├── tests/
│   └── test_optimizer.py         # 23 unit tests across 3 test classes
├── dashboard.py                  # Streamlit multi-tab frontend
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/madhavcodes25/ipl-moneyball.git
cd ipl-moneyball
```

### 2. Set up environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3) users:** The bundled PuLP CBC binary may be x86-only. Install a native solver:
> ```bash
> brew install cbc
> ```
> The app auto-detects and prefers the system CBC binary.

### 3. Launch the dashboard

```bash
streamlit run dashboard.py
```

### 4. Run the test suite

```bash
pytest tests/ -v
```

---

## 📊 Dashboard Tabs

### ⚙️ Team Optimizer
Configure budget, strategy, role counts, and player inclusions/exclusions. Click **Draft Perfect Team** to run the LP solver and get the mathematically optimal squad with analytics, a value efficiency chart, and CSV export.

### 📈 Market Analysis
The Moneyball view — an interactive scatter plot of every player's price vs. fantasy performance, overlaid with the **efficient frontier**. Includes:
- **Hidden Gems** — top 10 highest Value Score players above median fantasy output
- **Overpaid Players** — the 10 worst value players at high prices
- **Role-level value distribution** — box plot showing which positions have the most market inefficiency

### 🔍 Player Scout
Full searchable, filterable player database sorted by Value Score. Filter by role, nationality, and minimum fantasy score. Heatmap coloring on key metrics for quick comparison.

---

## 🧪 Test Coverage

```
tests/test_optimizer.py  —  24 tests

TestCalculateFantasy      (10)  score columns exist, non-negative, strategy weights, normalisation bounds
TestOptimizeTeam           (9)  11 players selected, budget respected, strict/flexible modes, inclusions/exclusions
TestGetEfficientFrontier   (5)  non-empty, subset of pool, sorted by cost, no dominated pairs
```

---

## 🤝 Contributors

- [Madhav Daluka](https://github.com/madhavcodes25/)
- [Sankalp Singh Chauhan](https://github.com/sankalpchauhan362-wq)
- [Harshraj Singh Panwar](https://github.com/HPisNotOP)
