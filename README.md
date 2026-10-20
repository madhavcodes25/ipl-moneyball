# 🏏 IPL Moneyball AI: Mathematical Team Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PuLP](https://img.shields.io/badge/Linear%20Programming-PuLP-orange)
![Tests](https://img.shields.io/badge/Tests-23%20passing-brightgreen)

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

$$\text{Fantasy Score} = \left(\text{norm\_Runs} \times 1.5 + \text{norm\_SR} \times 0.75 + \text{norm\_Avg} \times 0.75\right) \times w_{\text{bat}} + \left(\text{norm\_Wkts} \times 1.5 + \text{norm\_Econ} \times 0.75 + \text{norm\_BowlAvg} \times 0.75\right) \times w_{\text{bowl}}$$

- Normalization is bounded by *qualified* players (≥100 runs or ≥5 wickets) to prevent tail-enders with inflated small-sample stats from gaming the metric.
- Players with < 30 runs or < 2 wickets have their secondary metrics penalized by 80% to further guard against sample noise.

### Step 2: Value Score (The Moneyball Metric)

$$\text{Value Score} = \frac{\text{Fantasy Score}}{\text{Auction Price (Cr)}}$$

This single number answers *"how much output do you get per crore spent?"* — the exact question Moneyball asks about baseball players. Players with high Value Score but low absolute cost are the auction targets.

### Step 3: Efficient Frontier

For each player, we check if any other player exists with **both higher Fantasy Score and lower cost**. If so, the player is *dominated* — they offer worse value. The remaining players form the **Pareto-optimal efficient frontier**: the rational picks at each price point.

### Step 4: Integer Linear Programming

Team selection is formulated as a **Binary ILP** with PuLP:

- **Objective:** Maximize $\sum_{i} x_i \cdot \text{Fantasy\_Score}_i$
- **Decision variables:** $x_i \in \{0, 1\}$ (1 = selected, 0 = not selected)
- **Constraints:**
  - $\sum x_i = 11$ (squad size)
  - $\sum x_i \cdot \text{Cost}_i \leq \text{Budget}$ (budget cap)
  - $\sum x_i \cdot \mathbb{1}[\text{foreign}] \leq 4$ (overseas limit)
  - At least 1 WK, ≥5 bowling options, ≥6 batting options (IPL composition rules)
  - Optional: forced inclusions, exclusions, exact role counts

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Core Language | Python 3.9+ |
| Data Manipulation | Pandas, NumPy |
| Optimization Engine | PuLP (CBC/GLPK solver) |
| Web Dashboard | Streamlit |
| Visualizations | Plotly Express, Plotly Graph Objects |
| Testing | pytest (23 unit tests) |

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
Configure budget, strategy, role counts, and player inclusions/exclusions. Click **Draft Perfect Team** to run the LP solver and get the mathematically optimal squad with analytics, value efficiency charts, and CSV export.

### 📈 Market Analysis
The Moneyball view. An interactive scatter plot of every player's price vs. fantasy performance, overlaid with the **efficient frontier**. Includes:
- **Hidden Gems table** — top 10 highest Value Score players above the median fantasy score
- **Overpaid Players table** — the 10 worst value players at high prices
- **Role-level value distribution** — box plot showing which positions have the most market inefficiency

### 🔍 Player Scout
Full searchable, filterable player database sorted by Value Score. Filter by role, nationality, and minimum fantasy score. Heatmap coloring on key metrics for quick comparison.

---

## 🧪 Test Coverage

```
tests/test_optimizer.py — 23 tests

TestCalculateFantasy    (10 tests)  — score columns exist, non-negative, strategy weights, normalization bounds
TestOptimizeTeam        (8 tests)   — 11 players selected, budget respected, inclusions/exclusions, infeasible case
TestGetEfficientFrontier (5 tests)  — non-empty, subset of pool, sorted by cost, no dominated pairs
```

---

## 🤝 Contributing

Fork the repo, submit pull requests, or open issues for bugs and feature suggestions. See the test suite for expected behavior of all core functions.

## 🤝 Contributors
* [Madhav Daluka](https://github.com/madhavcodes25/)
* [Sankalp Singh Chauhan](https://github.com/sankalpchauhan362-wq)
* [Harshraj Singh Panwar](https://github.com/HPisNotOP)