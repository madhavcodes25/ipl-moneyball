# 🏏 IPL Moneyball: Mathematical Team Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Data Science](https://img.shields.io/badge/Data-Science-orange)

A data-driven, mathematically optimized Indian Premier League (IPL) team selector. Built for our Python data analysis group project, this dashboard applies linear programming to draft the ultimate fantasy squad based on 2025 player statistics.

## 🎯 Project Overview
Taking inspiration from *Moneyball*, this tool removes human bias from the auction table. By defining specific team constraints (like budget limits and foreign player caps) and dynamic strategies (Batting Heavy vs. Bowling Heavy), the app uses the `PuLP` library to solve a constrained knapsack problem, guaranteeing the highest possible fantasy score for the available budget.

## 🛠️ Tech Stack & Libraries
* **Core Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Mathematical Optimization:** PuLP (Linear Programming)
* **Web Interface:** Streamlit
* **Data Visualization:** Plotly Express, Matplotlib

## 🧠 Optimization Constraints
Our algorithm maximizes the "Total Fantasy Score" while strictly satisfying these real-world IPL rules:
* **Squad Size:** Exactly 11 players.
* **Budget:** Must stay under the user-defined limit (e.g., ₹100 Cr).
* **Overseas Players:** Maximum of 4 foreign players.
* **Wicketkeepers:** At least 1 WK.
* **Batting Depth:** Minimum of 6 recognized batting options.
* **Bowling Depth:** Minimum of 5 recognized bowling options.

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
