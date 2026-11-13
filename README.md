# 🏏 IPL Moneyball AI: Mathematical Team Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PuLP](https://img.shields.io/badge/Linear%20Programming-PuLP-orange)
![Tests](https://img.shields.io/badge/Tests-55%20passing-brightgreen)
![Players](https://img.shields.io/badge/Player%20Pool-381%20players-blue)
![Prices](https://img.shields.io/badge/Prices-IPL%202026%20Auction-gold)

> *"It's about getting things down to one number. Using the stats the way we read them, we'll find value in players that nobody else can see."* — Moneyball (2011)

A data-driven IPL team optimizer that applies **Integer Linear Programming**, **Pareto Efficiency Analytics**, and **Phase Specialist Scoring** (Powerplay, Middle, and Death overs) to identify undervalued talent and draft the highest-performing squad within a salary cap — inspired directly by Billy Beane's Moneyball philosophy.

---

## 🎯 The Core Problem

Traditional IPL auction strategy is driven by reputation, hype, recency bias, and intuition. Teams frequently overpay for big-name stars while overlooking high-performing, cost-effective players who produce equivalent or superior output at a fraction of the cost.

**IPL Moneyball solves this systematically through a two-stage mathematical pipeline:**

1. **Market & Value Analytics** — Scans 381 IPL players using recency-weighted ball-by-ball stats (IPL 2022–2026) and latest auction prices. Computes objective Fantasy Scores, evaluates Moneyball Value efficiency (output per crore), and surfaces the **Pareto-optimal efficient frontier** of non-dominated auction picks.
2. **Constrained Optimization Engine** — Formulates starting XI selection as a **Binary Integer Linear Program (ILP)** solved with `PuLP` and the `COIN-OR CBC` branch-and-cut solver. Guarantees a mathematically optimal team maximizing fantasy output under real IPL auction purse limits, overseas player restrictions, role quotas, pace/spin bowling distributions, and optional phase specialist constraints.

---

## 📐 Mathematical Formulation & Analytics Architecture (End-to-End)

This section details how every statistic, rating, composite index, and optimization constraint is computed from the ground up.

```
┌────────────────────────────────────────────────────────┐
│  Raw IPL Ball-by-Ball Data (87,687 deliveries, 2022-26) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Stage 1: 5-Season Recency Weighting & Aggregation     │
│  Linear season weights: w(2022)=1 ... w(2026)=5        │
└────────────┬─────────────────────────────┬─────────────┘
             │                             │
             ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ Stage 2: Phase Analytics  │ │ Stage 3: Core Career Norms│
│ PP, Middle, Death splits  │ │ Bounds on qualified pool  │
│ Bat & Bowl composites     │ │ 80% shrinkage penalties   │
└────────────┬──────────────┘ └────────────┬──────────────┘
             │                             │
             │ (Specialist Tags)           │ (Normalized Stats)
             │                             ▼
             │              ┌────────────────────────────────────────┐
             │              │  Stage 4: Composite Fantasy Rating     │
             │              │  Fantasy Score = Batting + Bowling     │
             │              │  (Grounded strictly in career output)  │
             │              └──────────────┬─────────────────────────┘
             │                             │
             │                             ▼
             │              ┌────────────────────────────────────────┐
             │              │  Stage 5: Moneyball Value Metric       │
             │              │  Value Score = Fantasy ÷ Price (₹ Cr)  │
             │              │  Pareto dominance -> Efficient Frontier│
             │              └──────────────┬─────────────────────────┘
             │                             │
             ▼                             ▼
┌────────────────────────────────────────────────────────┐
│  Stage 6: Integer Linear Programming (PuLP / CBC)      │
│  Maximize Sum(Fantasy Score) subject to IPL rules,     │
│  budget cap, WK, overseas, pace/spin, phase conditions │
└────────────────────────────────────────────────────────┘
```

---

### Stage 1: Ball-by-Ball Aggregation & 5-Season Recency Weighting

To capture current player form while preserving historical sample depth, career statistics are computed from **87,687 individual deliveries** across IPL seasons $s \in \{2022, 2023, 2024, 2025, 2026\}$.

A linear weighting function assigns increasing weight to more recent seasons:

$$
w_s = s - 2021 \quad \implies \quad w_{2022}=1,\; w_{2023}=2,\; w_{2024}=3,\; w_{2025}=4,\; w_{2026}=5
$$

A delivery bowled or faced in 2026 has **$5\times$ the statistical influence** of one in 2022.

#### 1. Career Batting Strike Rate
Weighted by ball volume per season:

$$
\text{SR}_{\text{career}} = \frac{\sum_{s} \text{Balls}_s \cdot \text{SR}_s \cdot w_s}{\sum_{s} \text{Balls}_s \cdot w_s}
$$

#### 2. Career Batting Average
Weighted by dismissals and runs:

$$
\text{Avg}_{\text{career}} = \frac{\sum_{s} \text{Runs}_s \cdot w_s}{\max\left(1, \; \sum_{s} \text{Dismissals}_s \cdot w_s\right)}
$$

#### 3. Career Bowling Economy Rate
Weighted by overs bowled ($\text{Balls}_s / 6$):

$$
\text{Econ}_{\text{career}} = \frac{\sum_{s} \text{RunsConceded}_s \cdot w_s}{\max\left(0.167, \; \sum_{s} \frac{\text{BallsBowled}_s}{6} \cdot w_s\right)}
$$

#### 4. Career Bowling Average
Weighted runs per wicket:

$$
\text{BowlAvg}_{\text{career}} = \frac{\sum_{s} \text{RunsConceded}_s \cdot w_s}{\max\left(1, \; \sum_{s} \text{Wickets}_s \cdot w_s\right)}
$$

---

### Stage 2: Phase-Specific Specialist Scoring

T20 matches have three distinct tactical phases with different requirements:
- **Powerplay (PP)**: Overs 1–6 (innings overs $0 \dots 5$) — field restrictions (max 2 fielders outside 30-yard circle).
- **Middle Overs (Mid)**: Overs 7–15 (innings overs $6 \dots 14$) — spin control, strike rotation, risk management.
- **Death Overs (Death)**: Overs 16–20 (innings overs $15 \dots 19$) — boundary hitting, yorkers, defensive variations.

#### Phase Qualification Thresholds
To prevent single-over anomalies from contaminating rankings:
- **Batting qualification**: $\ge 30$ balls (PP), $\ge 20$ balls (Middle), $\ge 15$ balls (Death)
- **Bowling qualification**: $\ge 24$ balls (PP), $\ge 30$ balls (Middle), $\ge 24$ balls (Death)

#### Phase Batting Composite
Evaluates speed, volume, and consistency in that exact phase:

$$
\text{Composite}_{\text{bat}, p} = 0.40 \cdot \text{norm}(\text{SR}_p) + 0.30 \cdot \text{norm}(\text{Runs}_p) + 0.30 \cdot \text{norm}(\text{Avg}_p)
$$

#### Phase Bowling Composite
Rewards both containment and breakthrough capability in that exact phase:
- Inverted economy (lower economy gives higher containment score):

$$
\text{norm}_{\text{inv}}(\text{Econ}_p) = \text{clip}\left(\frac{\max(\text{Econ}_p) - \text{Econ}_p}{\max(\text{Econ}_p) - \min(\text{Econ}_p) + 10^{-6}}, \; 0, \; 1\right)
$$

- Phase wickets (raw wicket taking in that specific phase):

$$
\text{norm}(\text{Wkts}_p) = \text{clip}\left(\frac{\text{Wkts}_p - \min(\text{Wkts}_p)}{\max(\text{Wkts}_p) - \min(\text{Wkts}_p) + 10^{-6}}, \; 0, \; 1\right)
$$

- Bowling Composite (50% phase wickets + 50% phase economy):

$$
\text{Composite}_{\text{bowl}, p} = 0.50 \cdot \text{norm}(\text{Wkts}_p) + 0.50 \cdot \text{norm}_{\text{inv}}(\text{Econ}_p)
$$

#### Aggregate Phase Score & Specialist Identification
The `Phase_Score` is the arithmetic mean across all available normalized phase metrics for a player, truncated to 3 decimal places:

$$
\text{Phase Score} = \text{round}\left(\frac{1}{|M_i|} \sum_{m \in M_i} m, \; 3\right)
$$

> **Design Choice**: The `Phase_Score` is **not blended into the overall Fantasy Score**. This keeps overall player valuation grounded purely in robust career production without distorting ratings for players with lower ball counts in specific phases. Instead, phase analytics are used strictly to:
> 1. **Rank specialists** in the Phase Analysis tab (Powerplay, Middle, Death).
> 2. **Tag top-quartile specialists** for the Integer Linear Program.
> 3. **Enforce tactical draft conditions** (e.g. guaranteeing at least 1 Death Overs batting specialist or 1 Powerplay bowler in the starting XI).

**Specialist Tags** are assigned to players ranking in the top quartile ($75^{\text{th}}$ percentile) of the respective phase composite:
- `pp_bat_specialist`: $\text{Composite}_{\text{bat}, \text{PP}} \ge Q_3$
- `death_bat_specialist`: $\text{Composite}_{\text{bat}, \text{Death}} \ge Q_3$
- `pp_bowl_specialist`: $\text{Composite}_{\text{bowl}, \text{PP}} \ge Q_3$
- `death_bowl_specialist`: $\text{Composite}_{\text{bowl}, \text{Death}} \ge Q_3$

---

### Stage 3: Min-Max Normalization & Small-Sample Shrinkage

Raw stats cannot be averaged directly due to differing scales and units (e.g. Strike Rate $\sim 140$ vs. Wickets $\sim 15$).

#### Robust Normalization Boundaries
To prevent non-specialist tail-enders from setting misleading boundaries, min/max thresholds $[\min_{\text{qual}}, \max_{\text{qual}}]$ are calculated **strictly from qualified players** ($\text{Runs} \ge 100$ for batting, $\text{Wickets} \ge 5$ for bowling):

- **Higher-is-better metrics** (Runs, Batting SR, Batting Avg, Wickets):

$$
\text{norm}(X) = \text{clip}\left(\frac{X - \min_{\text{qual}}(X)}{\max_{\text{qual}}(X) - \min_{\text{qual}}(X) + 10^{-6}}, \; 0, \; 1\right)
$$

- **Lower-is-better metrics** (Economy Rate, Bowling Avg):

$$
\text{norm}_{\text{inv}}(Y) = \text{clip}\left(\frac{\max_{\text{qual}}(Y) - Y}{\max_{\text{qual}}(Y) - \min_{\text{qual}}(Y) + 10^{-6}}, \; 0, \; 1\right)
$$

#### Small-Sample Shrinkage Penalties
To guard against small-sample noise (e.g. a batter who faced 2 balls and hit a six having an inflated 300.0 Strike Rate):

- **Batters with fewer than 30 runs** (`Runs < 30`): Normalized Strike Rate and Average are penalized by 80%:
  - $\text{norm(SR)} \leftarrow 0.20 \times \text{norm(SR)}$
  - $\text{norm(Avg)} \leftarrow 0.20 \times \text{norm(Avg)}$

- **Bowlers with fewer than 2 wickets** (`Wickets < 2`): Normalized Economy and Bowling Average are penalized by 80%:
  - $\text{norm(Econ)} \leftarrow 0.20 \times \text{norm(Econ)}$
  - $\text{norm(BowlAvg)} \leftarrow 0.20 \times \text{norm(BowlAvg)}$

---

### Stage 4: Composite Fantasy Rating

Each player receives a holistic `Fantasy_Score` derived from their career batting and bowling contributions, scaled by the chosen strategic focus:

#### 1. Batting Sub-Score

$$
\text{Score}_{\text{bat}} = 1.50 \cdot \text{norm}(\text{Runs}) + 0.75 \cdot \text{norm}(\text{SR}) + 0.75 \cdot \text{norm}(\text{Avg})
$$

#### 2. Bowling Sub-Score

$$
\text{Score}_{\text{bowl}} = 1.50 \cdot \text{norm}(\text{Wkts}) + 0.75 \cdot \text{norm}_{\text{inv}}(\text{Econ}) + 0.75 \cdot \text{norm}_{\text{inv}}(\text{BowlAvg})
$$

#### 3. Total Fantasy Score

$$
\text{Fantasy Score} = w_{\text{bat}} \cdot \text{Score}_{\text{bat}} + w_{\text{bowl}} \cdot \text{Score}_{\text{bowl}}
$$

where strategy weights adjust team focus:
- **Balanced**: $w_{\text{bat}} = 1.0, \quad w_{\text{bowl}} = 1.0$
- **Batting Heavy**: $w_{\text{bat}} = 1.2, \quad w_{\text{bowl}} = 0.8$
- **Bowling Heavy**: $w_{\text{bat}} = 0.8, \quad w_{\text{bowl}} = 1.2$

---

### Stage 5: Moneyball Value Analytics & Pareto Efficient Frontier

#### Price Resolution Hierarchy
1. **IPL 2026 Auction & Retentions** (Dec 2025 Abu Dhabi auction & 2025–2027 active contracts): Verified real prices for 264 players (Pant ₹27.0 Cr, Shreyas Iyer ₹26.75 Cr, Green ₹25.2 Cr, Klaasen ₹23.0 Cr, Kohli ₹21.0 Cr, Bumrah ₹18.0 Cr, Ruturaj Gaikwad ₹18.0 Cr, Samson ₹18.0 Cr, Cummins ₹18.0 Cr, etc.).
2. **Legacy Auction History**: Previous IPL auction price if no 2026 auction record exists.
3. **Role-Based Median Estimate**: Unsold or new overseas players receive the median price of their respective role.

#### The Value Score (Moneyball Efficiency Metric)

$$
\text{Value Score} = \frac{\text{Fantasy Score}}{\text{Auction Price (₹ Cr)}}
$$

This represents **fantasy points generated per crore spent**. High-output players available at modest valuations are highlighted as **Hidden Gems**.

#### Pareto Dominance & Efficient Frontier
In multi-objective optimization, a player $A$ **Pareto-dominates** player $B$ if:

$$
\text{Fantasy Score}(A) \ge \text{Fantasy Score}(B) \quad \text{AND} \quad \text{Cost}(A) \le \text{Cost}(B)
$$

with at least one strict inequality.

Players who are not dominated by any other player form the **Pareto Efficient Frontier**: the set of mathematically optimal player choices at every price point in the auction.

---

### Stage 6: Constrained Integer Linear Programming (PuLP / CBC)

Team drafting is formulated as a **Binary Integer Linear Program (ILP)**:

Let $x_i \in \{0, 1\}$ be the decision variable indicating whether player $i \in \{1, \dots, n\}$ is drafted ($x_i = 1$) or not ($x_i = 0$).

#### Objective Function

$$
\max_{x} \sum_{i=1}^{n} x_i \cdot \text{Fantasy Score}_i
$$

The solver maximizes the aggregate fantasy rating of the starting XI subject to the following system of linear constraints:

#### General Auction & Composition Constraints
| Constraint | Mathematical Formulation | Description |
|---|---|---|
| **Binary selection** | $x_i \in \{0, 1\}$ | $x_i = 1$ if player $i$ is drafted, $0$ otherwise |
| **Squad size** | $\sum_{i=1}^{n} x_i = 11$ | Exactly 11 players in the starting XI |
| **Budget cap** | $\sum_{i=1}^{n} x_i \cdot \text{Cost}_i \le \text{Budget}$ | Total expenditure within auction purse (e.g. ₹100 Cr) |
| **Overseas limit** | $\sum_{i \in \text{Foreign}} x_i \le 4$ | Maximum of 4 foreign players in the starting XI |
| **Wicketkeeper** | $\sum_{i \in \text{WK}} x_i \ge 1$ | At least 1 designated wicketkeeper |

#### Bowling Style & Role Composition Constraints (Strict Mode)
| Constraint | Mathematical Formulation | Description |
|---|---|---|
| **Pace Bowling Attack** | $\sum_{i \in \text{Pacer}} x_i \ge N_{\text{pacers}}$ | At least $N_{\text{pacers}}$ pace bowlers (default $\ge 3$) |
| **Spin Bowling Attack** | $\sum_{i \in \text{Spinner}} x_i \ge N_{\text{spinners}}$ | At least $N_{\text{spinners}}$ spin bowlers (default $\ge 2$) |
| **Exact Batsmen** | $\sum_{i \in \text{Batsman}} x_i = N_{\text{batters}}$ | Exact count of pure batsmen (default 4) |
| **Exact Bowlers** | $\sum_{i \in \text{Bowler}} x_i = N_{\text{bowlers}}$ | Exact count of pure bowlers (default 4) |
| **Exact All-Rounders** | $\sum_{i \in \text{All-Rounder}} x_i = N_{\text{all}}$ | Exact count of all-rounders (default 3) |

#### Flexible Mode Composition Constraints
| Constraint | Mathematical Formulation | Description |
|---|---|---|
| **Batting depth** | $\sum_{i \in \text{Bat/AllR}} x_i \ge 6$ | Minimum 6 recognized batting options |
| **Bowling depth** | $\sum_{i \in \text{Bowl/AllR}} x_i \ge 5$ | Minimum 5 recognized bowling options |

#### Phase Specialist Constraints *(Optional User Toggles)*
| Constraint | Mathematical Formulation | Description |
|---|---|---|
| **PP Bat Specialist** | $\sum_{i \in \text{PP Bat}} x_i \ge 1$ | Forces at least 1 top-quartile Powerplay batter |
| **Death Bat Specialist** | $\sum_{i \in \text{Death Bat}} x_i \ge 1$ | Forces at least 1 top-quartile Death Overs batter |
| **PP Bowl Specialist** | $\sum_{i \in \text{PP Bowl}} x_i \ge 1$ | Forces at least 1 top-quartile Powerplay bowler |
| **Death Bowl Specialist** | $\sum_{i \in \text{Death Bowl}} x_i \ge 1$ | Forces at least 1 top-quartile Death Overs bowler |

#### Mandatory Inclusions & Exclusions
- **Retained / Locked Players**: $x_k = 1 \quad \forall k \in \text{MustInclude}$
- **Banned / Unavailable Players**: $x_j = 0 \quad \forall j \in \text{MustExclude}$

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Core Language | Python 3.9+ |
| Data Manipulation | Pandas, NumPy |
| Optimization Engine | PuLP (COIN-OR CBC / GLPK solver) |
| Web Dashboard | Streamlit |
| Visualizations | Plotly Express, Plotly Graph Objects |
| Ball-by-ball Data | RR Hackathon Dataset (87,687 deliveries, IPL 2022–2026) |
| Pricing | IPL 2026 Auction & Retentions (Dec 2025) |
| Testing | pytest — 53 unit tests |

---

## 📂 Repository Structure

```text
IPL-Moneyball/
├── data/
│   ├── raw/
│   │   ├── IPL_Stats.csv                      # Legacy player stats (fallback only)
│   │   └── IPL_2026_prices.csv                # IPL 2026 retention & auction prices (300 players)
│   ├── processed/
│   │   └── bbb_derived_stats.csv              # 381 players — BBB-derived, recency-weighted
│   └── RR-Hackathon-Data/
│       ├── ball_by_ball.csv                   # 200,999 ball-by-ball deliveries (all formats)
│       ├── matches.csv                        # Match metadata with competition IDs
│       ├── competitions.csv                   # Competition list (IPL comp IDs: 1358–3298)
│       ├── teams.csv
│       ├── playing_eleven.csv
│       └── competition_team_player_mapping.csv
├── src/
│   ├── bbb_data_generator.py                  # Generates bbb_derived_stats.csv from BBB data
│   ├── data_cleaning.py                       # Data loading, dedup, phase stats merge
│   ├── optimizer.py                           # Fantasy scoring, Phase Score, LP model, frontier
│   └── phase_stats.py                         # Phase specialist stats from ball-by-ball data
├── tests/
│   └── test_optimizer.py                      # 53 unit tests across 6 test classes
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

### 3. (Optional) Regenerate the player pool from BBB data

```bash
python -m src.bbb_data_generator
```

### 4. Launch the dashboard

```bash
streamlit run dashboard.py
```

### 5. Run the test suite

```bash
pytest tests/ -v
```

---

## 📊 Dashboard Tabs

### ⚙️ Team Optimizer
Configure budget, strategy, role counts, pace/spin requirements, and player inclusions/exclusions. Click **Draft Perfect Team** to run the LP solver and view:
- **Total Fantasy Points KPI**: The maximized objective value.
- **Budget & Value Analytics**: Total spend, remaining funds, foreign player counts, and squad value efficiency.
- **Visual Analytics**: Interactive role expenditure pie chart and player fantasy contribution bar chart.
- **Optimal Squad Roster**: Comprehensive table with player roles, price year, last season, phase badges, and one-click CSV export.

### 📈 Market Analysis
The Moneyball view — an interactive scatter plot of every player's price vs. fantasy performance, overlaid with the **efficient frontier**:
- **Hidden Gems** — top 10 highest Value Score players filtered by a minimum fantasy score threshold to highlight genuine impact players.
- **Overpaid Players** — high-priced players with poor fantasy return.
- **Role-level value distribution** — box plot showing positional market inefficiency.

### 🔍 Player Scout
Full searchable, filterable player database. Sort dynamically by **Fantasy Score**, **Value Score**, **Price**, **Batting Runs**, or **Wickets**. Includes nationality, role, and minimum score filters with heatmap color coding.

### ⚡ Phase Analysis
Ball-by-ball specialist analysis using **IPL 2022–2026 data** with recency-weighted multi-year averages:
- **Phase selector** — switch between Powerplay, Middle Overs, and Death Overs views.
- **Batting specialists** — ranked by phase composite (SR, Runs, and Avg).
- **Bowling specialists** — ranked by phase composite (50% phase wickets + 50% economy containment).
- **Phase heatmap** — multi-phase comparison grid for the top 30 players.
- **All-Phase summary table** — complete matrix of PP, Middle, and Death metrics in one view.

---

## 🧪 Test Coverage

```
tests/test_optimizer.py  —  55 tests

TestCalculateFantasy          (10)  score columns, non-negative, strategy weights, normalisation
TestOptimizeTeam               (9)  11 players, budget, strict/flexible modes, inclusions/exclusions
TestGetEfficientFrontier       (5)  non-empty, subset of pool, sorted, no dominated pairs
TestLoadPhaseStats             (8)  DataFrame structure, Pakistani filter, SR/economy validity, no dupes
TestCalculateFantasyWithPhase  (8)  Phase_Score column, 3-decimal truncation, specialist tags, independence
TestPhaseSpecialistConstraints (4)  LP constraints for PP bat/bowl, Death bowl, all 4 together
TestBBBDataGenerator          (11)  returns DataFrame, 381 players, 2026 prices, recency weights,
                                    alias map, role inference, no wkts_per_over columns
```

---

## 🤝 Contributors

- [Madhav Daluka](https://github.com/madhavcodes25/)
- [Sankalp Singh Chauhan](https://github.com/sankalpchauhan362-wq)
- [Harshraj Singh Panwar](https://github.com/HPisNotOP)
