# 🏏 IPL Moneyball: Mathematical Team Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Data Science](https://img.shields.io/badge/Data-Science-orange)

A data-driven, mathematically optimized Indian Premier League (IPL) squad selector. This dashboard applies Linear Programming to draft the ultimate fantasy team based on 2025 player statistics, allowing users to play the role of a Head Coach with a data-science edge.

## 🎯 Project Overview
Taking inspiration from *Moneyball*, this tool removes human bias from the auction table. By defining specific team constraints (like budget limits and foreign player caps) and dynamic strategies (Batting Heavy vs. Bowling Heavy), the app uses the `PuLP` library to solve a constrained knapsack problem, guaranteeing the highest possible fantasy score for the available budget.

## 🛠️ Tech Stack & Libraries
* **Core Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Mathematical Optimization:** PuLP (Linear Programming)
* **Web Interface:** Streamlit
* **Data Visualization:** Plotly Express

## ✨ Advanced Features
* **Intelligent Retention System**: Force include specific marquee players (e.g., Kohli, Bumrah) into your squad. The optimizer automatically adjusts the remaining budget and roster slots around your core picks.
* **Ignore Player System:**: Force exclude specific players into your squad.
* **Dynamic Role Configuration:** Unlike standard selectors, you can define your own team balance.
* **Strategy Weighting:** Toggle between "Batting Heavy," "Bowling Heavy," or "Balanced" optimization modes to align with specific ground conditions (e.g., Chinnaswamy vs. Chepauk).

### 🧠 Optimization Logic (The Math)

The core engine of this project treats IPL squad selection as a **Constrained Knapsack Problem**. We use Linear Programming to find the global optimum rather than relying on a simple greedy search.

#### 1. Objective Function
We aim to maximize the total projected fantasy potential ($Z$) of the squad:
$$\text{Maximize } Z = \sum_{i=1}^{n} (\text{Fantasy Score}_i \times x_i)$$
Where $x_i$ is a binary decision variable: $1$ if player $i$ is drafted, $0$ otherwise.

#### 2. Key Constraints
The solver must satisfy the following linear inequalities:

* **Squad Size:** Exactly 11 players must be selected.
  $$\sum_{i=1}^{n} x_i = 11$$

* **Budget Constraint:** Total cost cannot exceed the user-defined limit ($C$).
  $$\sum_{i=1}^{n} (\text{Cost}_i \times x_i) \leq C$$

* **Overseas Cap:** Maximum of 4 foreign players.
  $$\sum_{i=1}^{n} (\text{IsForeign}_i \times x_i) \leq 4$$

* **Role Balance:** Ensures the squad meets the specific counts for Batters, Bowlers, and All-rounders defined in the dashboard.
  $$\sum_{i \in \text{Role}_j} x_i = \text{Target Count}_j$$

## 📂 Repository Structure

```text
├── data/
│   └── raw/
│       └── IPL_Stats.csv         # Raw player performance stats
├── src/
│   ├── data_cleaning.py          # Data preprocessing and metric normalization
│   └── optimizer.py              # LP model and objective function definition
├── dashboard.py                  # Streamlit frontend app
├── README.md                     
└── requirements.txt
```

## 🚀 How to Run Locally

1. **Clone the repo:**
   ```Bash
   git clone https://github.com/madhavcodes25/ipl-moneyball.git
   cd ipl-moneyball
   ```
   
2. **Install dependencies:**

```Bash
pip install streamlit pandas numpy pulp plotly matplotlib
```

3. **Launch the dashboard:**

```Bash
streamlit run dashboard.py
```

Feel free to fork this project, submit pull requests, or open issues if you find any bugs or have feature suggestions!
