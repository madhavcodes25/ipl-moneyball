# 🏏 IPL Moneyball AI: Mathematical Team Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PuLP](https://img.shields.io/badge/Linear%20Programming-PuLP-orange)
![Tests](https://img.shields.io/badge/Tests-38%20passing-brightgreen)

> *"It's about getting things down to one number. Using the stats the way we read them, we'll find value in players that nobody else can see."* — Moneyball (2011)

A data-driven IPL team optimizer that applies **Linear Programming**, **Value Analytics**, and **Phase Specialist Scoring** (Powerplay / Middle / Death overs) to identify undervalued players and construct the highest-performing squad within a given auction budget — inspired directly by the Moneyball philosophy.

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

$$
\text{Fantasy Score} = \Big(\text{Runs}_{\text{norm}} \times 1.5 + \text{SR}_{\text{norm}} \times 0.75 + \text{Avg}_{\text{norm}} \times 0.75\Big) \cdot w_{\text{bat}} + \Big(\text{Wkts}_{\text{norm}} \times 1.5 + \text{Econ}_{\text{norm}} \times 0.75 + \text{BowlAvg}_{\text{norm}} \times 0.75\Big) \cdot w_{\text{bowl}} + \text{Phase Score} \times 0.5
$$

- All metrics are min-max scaled to $[0, 1]$ using only **qualified** players ($\ge 100$ runs or $\ge 5$ wickets) as boundaries.
- Players with fewer than 30 runs or 2 wickets have their secondary metrics penalized by 80% to guard against small-sample noise.
- Strategy weights: $w_{\text{bat}} = w_{\text{bowl}} = 1.0$ (Balanced), $1.2 / 0.8$ (Batting Heavy), $0.8 / 1.2$ (Bowling Heavy).

### Step 1b: Phase Specialist Score (New)

Using **ball-by-ball data from IPL 2022–2026** (87,687 deliveries across 5 seasons), each player receives phase-specific performance metrics with **recency-weighted multi-year averages**:

$$
\text{Weighted SR} = \frac{\sum_{s} \text{balls}_s \cdot \text{SR}_s \cdot w_s}{\sum_{s} \text{balls}_s \cdot w_s}
$$

where season weights are $w_{2022}=1,\ w_{2023}=2,\ w_{2024}=3,\ w_{2025}=4,\ w_{2026}=5$ — giving more recent seasons greater influence.

| Phase | Overs | Batting metric | Bowling metric | Min balls (bat/bowl) |
|---|---|---|---|---|
| Powerplay | 1–6 | Strike Rate | Economy | 30 / 12 |
| Middle | 7–15 | Strike Rate | Economy | 20 / 18 |
| Death | 16–20 | Strike Rate | Economy | 15 / 12 |

The `Phase_Score` is the average of all available phase norms (batting SR: higher = better; bowling economy: lower = better, inverted). It is blended into `Fantasy_Score` at a **0.5× weight** — it nudges but does not dominate the core stats-based score.

**Specialist tags** (`pp_bat_specialist`, `death_bat_specialist`, `pp_bowl_specialist`, `death_bowl_specialist`) are assigned to players in the **top/bottom quartile** of their respective phase metric.

### Step 2: Value Score (The Moneyball Metric)

$$
\text{Value Score} = \frac{\text{Fantasy Score}}{\text{Auction Price (Cr)}}
$$

This single number answers *"how much output do you get per crore spent?"* — the exact question Moneyball asks. Players with a high Value Score but low absolute cost are the auction targets.

### Step 3: Efficient Frontier

For each player, we check whether any other player exists with **both a higher Fantasy Score and a lower cost**. If so, the player is *dominated*. The remaining players form the **Pareto-optimal efficient frontier**: the rational picks at each price point.

### Step 4: Integer Linear Programming

Team selection is formulated as a **Binary Integer Linear Program (ILP)** solved by PuLP + CBC:

$$
\max \sum_{i=1}^{n} x_i \cdot \text{Fantasy Score}_i
$$

Subject to the following constraints:

| Constraint | Mathematical Formulation | Description |
|---|---|---|
| **Binary selection** | $x_i \in \{0, 1\}$ | $x_i = 1$ if player $i$ is drafted, $0$ otherwise |
| **Squad size** | $\sum_{i=1}^{n} x_i = 11$ | Exactly 11 players in the starting XI |
| **Budget cap** | $\sum_{i=1}^{n} x_i \cdot \text{Cost}_i \le \text{Budget}$ | Total expenditure within auction purse |
| **Overseas limit** | $\sum_{i=1}^{n} x_i \cdot \text{Overseas}_i \le 4$ | Maximum of 4 foreign players |
| **Wicketkeeper** | $\sum_{i \in \text{WK}} x_i \ge 1$ | At least 1 designated wicketkeeper |
| **Batting depth** | $\sum_{i \in \text{Bat/AllR}} x_i \ge 6$ | Minimum 6 recognized batting options |
| **Bowling depth** | $\sum_{i \in \text{Bowl/AllR}} x_i \ge 5$ | Minimum 5 recognized bowling options |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Core Language | Python 3.9+ |
| Data Manipulation | Pandas, NumPy |
| Optimization Engine | PuLP (CBC / GLPK solver) |
| Web Dashboard | Streamlit |
| Visualizations | Plotly Express, Plotly Graph Objects |
| Ball-by-ball Data | RR Hackathon Dataset (IPL 2022–2026) |
| Testing | pytest — 38 unit tests |

---

## 📂 Repository Structure

```text
IPL-Moneyball/
├── data/
│   ├── raw/
│   │   └── IPL_Stats.csv                      # 2024–2025 IPL player stats & auction prices
│   └── RR-Hackathon-Data/
│       ├── ball_by_ball.csv                   # 200,999 ball-by-ball deliveries (all formats)
│       ├── matches.csv                        # Match metadata with competition IDs
│       ├── competitions.csv                   # Competition list (IPL comp IDs: 1358–3298)
│       ├── teams.csv
│       ├── playing_eleven.csv
│       └── competition_team_player_mapping.csv
├── src/
│   ├── data_cleaning.py                       # Data loading, dedup, phase stats merge
│   ├── optimizer.py                           # Fantasy scoring, Phase Score, LP model, frontier
│   └── phase_stats.py                         # Phase specialist stats from ball-by-ball data
├── tests/
│   └── test_optimizer.py                      # 38 unit tests across 5 test classes
├── dashboard.py                               # Streamlit multi-tab frontend
├── requirements.txt
├── packages.txt                               # System packages for Streamlit Cloud (coinor-cbc)
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

### ⚡ Phase Analysis *(New)*
Ball-by-ball specialist analysis using **IPL 2022–2026 data** with recency-weighted multi-year averages:
- **Phase selector** — switch between Powerplay, Middle Overs, and Death Overs views
- **Batting specialists** — ranked by phase-specific strike rate with minimum ball thresholds
- **Bowling specialists** — ranked by phase-specific economy (lower = better)
- **Phase heatmap** — multi-phase comparison grid for the top 30 players by Fantasy Score
- **All-Phase summary table** — every player's PP / Mid / Death SR and economy in one view

---

## 🧪 Test Coverage

```
tests/test_optimizer.py  —  38 tests

TestCalculateFantasy          (10)  score columns, non-negative, strategy weights, normalisation
TestOptimizeTeam               (9)  11 players, budget, strict/flexible modes, inclusions/exclusions
TestGetEfficientFrontier       (5)  non-empty, subset of pool, sorted, no dominated pairs
TestLoadPhaseStats             (8)  DataFrame structure, Pakistani filter, SR/economy validity, no dupes
TestCalculateFantasyWithPhase  (6)  Phase_Score column, specialist tags, non-negative, no NaN
```

---

## 🤝 Contributors

- [Madhav Daluka](https://github.com/madhavcodes25/)
- [Sankalp Singh Chauhan](https://github.com/sankalpchauhan362-wq)
- [Harshraj Singh Panwar](https://github.com/HPisNotOP)
