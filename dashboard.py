import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_cleaning import prepare_data
from src.optimizer import calculate_fantasy, optimize_team, get_efficient_frontier

st.set_page_config(page_title="IPL Moneyball", page_icon="🏏", layout="wide")

@st.cache_data
def load_data():
    return prepare_data()

def render_sidebar(player_list, raw_data):
    """Renders the Strategy Room sidebar and returns all user-configured parameters."""
    with st.sidebar:
        st.header("Strategy Room 🏏")
        st.markdown("Fine-tune your auction constraints.")

        constraint_mode = st.radio(
            "Select Constraint Mode",
            ["Flexible (Auto-balance)", "Strict (Custom Roles)"],
            help="Flexible ensures min 5 bowlers and 6 batters. Strict lets you pick exact counts."
        )

        budget = st.slider("Total Budget (Crores)", 20, 120, 100)
        role_focus = st.selectbox("Team Strategy", ["Balanced", "Batting Heavy", "Bowling Heavy"])

        st.markdown("---")
        st.subheader("Team Composition")

        num_batters, num_bowlers, num_all_rounders = 5, 4, 2
        num_pacers, num_spinners = 3, 2

        if constraint_mode == "Strict (Custom Roles)":
            st.caption("Must exactly equal 11 players")
            col1, col2 = st.columns(2)
            with col1:
                num_batters = st.number_input("Batters 🏏", min_value=1, max_value=11, value=5)
            with col2:
                num_bowlers = st.number_input("Bowlers 🎯", min_value=0, max_value=11, value=4)

            num_all_rounders = st.number_input("All-Rounders ⚔️", min_value=0, max_value=11, value=2)

            st.subheader("Bowling Attack Setup")
            col3, col4 = st.columns(2)
            with col3:
                num_pacers = st.number_input("Min Pacers", min_value=0, max_value=11, value=3)
            with col4:
                num_spinners = st.number_input("Min Spinners", min_value=0, max_value=11, value=2)
        else:
            st.info(
                "🤖 **Auto-balance Active**\n\n"
                "The algorithm will automatically build the most optimal squad ensuring at least:\n"
                "- 6 Batting options\n"
                "- 5 Bowling options\n"
                "- 1 Wicket Keeper"
            )

        if (num_batters + num_bowlers + num_all_rounders) != 11:
            st.error("🚨 Constraint Error: The sum of Batters, Bowlers, and All-Rounders must equal 11.")
            return None

        st.markdown("---")
        st.subheader("Retentions")
        must_include_players = st.multiselect(
            "Force include up to 11 players:",
            options=player_list,
            max_selections=11,
            help="Will build the rest of the team around these retained players."
        )

        retention_prices = {}
        if must_include_players:
            st.caption("Set custom retention prices (₹ Cr)")
            for player in must_include_players:
                default_cost = float(raw_data[raw_data['Player Name'] == player]['Cost_Cr'].iloc[0])
                retention_prices[player] = st.number_input(
                    f"{player} Price",
                    min_value=0.0,
                    max_value=float(budget),
                    value=default_cost,
                    step=0.25
                )

        st.markdown("---")
        st.subheader("Exclusions")
        must_exclude_players = st.multiselect(
            "Ignore these players:",
            options=player_list,
            help="These players will be completely ignored by the optimizer."
        )
        st.markdown("---")

        with st.expander("🛠️ Custom Expected Prices"):
            st.caption("Adjust expected prices for players in the auction pool.")
            available_for_custom_price = [
                p for p in player_list
                if p not in must_include_players and p not in must_exclude_players
            ]
            players_to_adjust = st.multiselect(
                "Select players to adjust:",
                options=available_for_custom_price,
                help="Override the dataset's default price for these players."
            )
            custom_auction_prices = {}
            if players_to_adjust:
                for player in players_to_adjust:
                    default_cost = float(raw_data[raw_data['Player Name'] == player]['Cost_Cr'].iloc[0])
                    custom_auction_prices[player] = st.number_input(
                        f"{player} Expected Price",
                        min_value=0.0,
                        max_value=float(budget),
                        value=default_cost,
                        step=0.25,
                        key=f"auction_price_{player}"
                    )
            else:
                custom_auction_prices = {}

        st.markdown("---")
        generate_btn = st.button("Draft Perfect Team", type="primary", use_container_width=True)

    return dict(
        budget=budget,
        role_focus=role_focus,
        constraint_mode=constraint_mode,
        num_batters=num_batters,
        num_bowlers=num_bowlers,
        num_all_rounders=num_all_rounders,
        num_pacers=num_pacers,
        num_spinners=num_spinners,
        must_include_players=must_include_players,
        must_exclude_players=must_exclude_players,
        retention_prices=retention_prices,
        custom_auction_prices=custom_auction_prices,
        generate_btn=generate_btn,
    )


def render_market_analysis_tab(scored_data: pd.DataFrame):
    """
    Renders the Market Analysis tab — the core Moneyball insight.
    Shows the Price vs. Fantasy Score scatter, the efficient frontier, and the Hidden Gems table.
    """
    st.subheader("📈 Price vs. Performance: The Moneyball Market")
    st.markdown(
        "The chart below maps every player's **auction price** against their **fantasy output**. "
        "The **Efficient Frontier** (dashed line) traces the players who represent the "
        "best possible performance at each price point — the market-rational picks. "
        "Players **below and to the right** of the frontier are overpaid. "
        "Players **above and to the left** are undervalued — the true Moneyball targets."
    )

    display_df = scored_data.dropna(subset=['Cost_Cr', 'Fantasy_Score']).copy()
    display_df = display_df[display_df['Cost_Cr'] > 0]

    frontier_df = get_efficient_frontier(display_df)

    # --- Scatter plot ---
    fig = px.scatter(
        display_df,
        x='Cost_Cr',
        y='Fantasy_Score',
        color='Role',
        size='Value_Score',
        hover_name='Player Name',
        hover_data={
            'Cost_Cr': ':.2f',
            'Fantasy_Score': ':.3f',
            'Value_Score': ':.3f',
            'Role': True,
            'Subrole': True,
        },
        labels={
            'Cost_Cr': 'Auction Price (₹ Cr)',
            'Fantasy_Score': 'Fantasy Score',
            'Value_Score': 'Value Score (Fantasy/₹Cr)',
        },
        title="Player Market Map: Price vs. Fantasy Performance",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.75,
        size_max=25,
    )

    # Overlay efficient frontier as a step line
    if not frontier_df.empty:
        fig.add_trace(go.Scatter(
            x=frontier_df['Cost_Cr'],
            y=frontier_df['Fantasy_Score'],
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='gold', width=2.5, dash='dash'),
            marker=dict(size=8, symbol='diamond', color='gold', line=dict(color='black', width=1)),
        ))

    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(14,17,23,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Hidden Gems & Overpaid sections ---
    col_gems, col_over = st.columns(2)

    with col_gems:
        st.subheader("💎 Hidden Gems (Most Undervalued)")
        st.caption("High fantasy output per crore spent — ideal auction targets.")
        gems = (
            display_df[display_df['Fantasy_Score'] > display_df['Fantasy_Score'].median()]
            .sort_values('Value_Score', ascending=False)
            .head(10)[['Player Name', 'Role', 'Cost_Cr', 'Fantasy_Score', 'Value_Score']]
            .reset_index(drop=True)
        )
        gems.index += 1
        gems.columns = ['Player', 'Role', 'Price (Cr)', 'Fantasy Score', 'Value Score']
        gems['Price (Cr)'] = gems['Price (Cr)'].map('₹{:.2f}'.format)
        gems['Fantasy Score'] = gems['Fantasy Score'].map('{:.3f}'.format)
        gems['Value Score'] = gems['Value Score'].map('{:.3f}'.format)
        st.dataframe(gems, use_container_width=True, height=380)

    with col_over:
        st.subheader("💸 Overpaid Players (Worst Value)")
        st.caption("High price tag but low fantasy returns — players to avoid at auction.")
        overpaid = (
            display_df[display_df['Cost_Cr'] >= display_df['Cost_Cr'].median()]
            .sort_values('Value_Score', ascending=True)
            .head(10)[['Player Name', 'Role', 'Cost_Cr', 'Fantasy_Score', 'Value_Score']]
            .reset_index(drop=True)
        )
        overpaid.index += 1
        overpaid.columns = ['Player', 'Role', 'Price (Cr)', 'Fantasy Score', 'Value Score']
        overpaid['Price (Cr)'] = overpaid['Price (Cr)'].map('₹{:.2f}'.format)
        overpaid['Fantasy Score'] = overpaid['Fantasy Score'].map('{:.3f}'.format)
        overpaid['Value Score'] = overpaid['Value Score'].map('{:.3f}'.format)
        st.dataframe(overpaid, use_container_width=True, height=380)

    st.markdown("---")

    # --- Role-based value distribution ---
    st.subheader("🎯 Value Score Distribution by Role")
    st.caption("Box plot showing the spread of Value Scores per role — useful for spotting which positions have the most inefficiency.")
    fig_box = px.box(
        display_df,
        x='Role',
        y='Value_Score',
        color='Role',
        points='all',
        hover_name='Player Name',
        labels={'Value_Score': 'Value Score (Fantasy/₹Cr)', 'Role': 'Player Role'},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_box.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(14,17,23,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    )
    st.plotly_chart(fig_box, use_container_width=True)


def render_player_scout_tab(scored_data: pd.DataFrame):
    """Renders an interactive player comparison and scouting table."""
    st.subheader("🔍 Player Scouting Database")
    st.caption("Search, filter, and sort all players by any metric. Great for live auction preparation.")

    col1, col2, col3 = st.columns(3)
    with col1:
        role_filter = st.multiselect("Filter by Role", options=sorted(scored_data['Role'].dropna().unique()), default=[])
    with col2:
        nationality_filter = st.radio("Nationality", ["All", "Indian", "Overseas"], horizontal=True)
    with col3:
        min_score = st.slider(
            "Min Fantasy Score",
            min_value=0.0,
            max_value=float(scored_data['Fantasy_Score'].max()),
            value=0.0,
            step=0.1,
        )

    scout_df = scored_data.copy()
    if role_filter:
        scout_df = scout_df[scout_df['Role'].isin(role_filter)]
    if nationality_filter == "Indian":
        scout_df = scout_df[scout_df['Is_Foreign'] == 0]
    elif nationality_filter == "Overseas":
        scout_df = scout_df[scout_df['Is_Foreign'] == 1]
    scout_df = scout_df[scout_df['Fantasy_Score'] >= min_score]

    display_cols = ['Player Name', 'Role', 'Subrole', 'Cost_Cr', 'Is_Foreign',
                    'Batting Runs', 'Batting Average', 'Batting Strike Rate',
                    'Wickets', 'Economy Rate', 'Fantasy_Score', 'Value_Score']
    scout_display = (
        scout_df[display_cols]
        .sort_values('Value_Score', ascending=False)
        .reset_index(drop=True)
    )
    scout_display.index += 1

    scout_display = scout_display.rename(columns={
        'Cost_Cr': 'Price (Cr)',
        'Is_Foreign': 'Overseas',
        'Fantasy_Score': 'Fantasy Score',
        'Value_Score': 'Value Score',
        'Batting Runs': 'Runs',
        'Batting Average': 'Bat Avg',
        'Batting Strike Rate': 'SR',
        'Economy Rate': 'Econ',
    })

    st.dataframe(
        scout_display.style.background_gradient(subset=['Fantasy Score', 'Value Score'], cmap='YlGn'),
        use_container_width=True,
        height=500,
    )
    st.caption(f"Showing {len(scout_display)} players. Sorted by Value Score (descending).")


def render_optimizer_tab(params: dict, raw_data: pd.DataFrame):
    """Runs the LP optimizer and renders team results."""
    generate_btn = params['generate_btn']

    if not generate_btn:
        st.info("👈 Adjust your parameters in the **Strategy Room** and click **'Draft Perfect Team'** to begin.")
        return

    must_include_players = params['must_include_players']
    must_exclude_players = params['must_exclude_players']
    budget = params['budget']

    conflict_players = set(must_include_players).intersection(set(must_exclude_players))
    if conflict_players:
        st.error(f"🚨 Logic Error: You cannot both retain and ignore: {', '.join(conflict_players)}")
        return

    if (params['num_batters'] + params['num_bowlers'] + params['num_all_rounders']) != 11:
        st.error("🚨 Constraint Error: Batters + Bowlers + All-Rounders must equal 11.")
        return

    if (params['num_pacers'] + params['num_spinners']) > (params['num_bowlers'] + params['num_all_rounders']):
        st.warning("⚠️ Logic Warning: More Pacers+Spinners requested than total bowling slots. Optimizer may fail.")

    if must_include_players:
        retained_df = raw_data[raw_data['Player Name'].isin(must_include_players)]
        foreign_col = 'Is_Foreign' if 'Is_Foreign' in retained_df.columns else 'Overseas'
        budget_spent_retained = sum(params['retention_prices'].values())
        overseas_count_retained = retained_df[foreign_col].sum() if foreign_col in retained_df.columns else 0

        if overseas_count_retained > 4:
            st.error(f"🚨 Rule Violation: {int(overseas_count_retained)} retained overseas players — max is 4.")
            return
        if budget_spent_retained > budget:
            st.error(f"🚨 Budget Exceeded: Retained players cost ₹{budget_spent_retained:.2f} Cr — over your ₹{budget} Cr limit.")
            return

    with st.spinner("Crunching the numbers and simulating auction bids..."):
        scored_data = calculate_fantasy(raw_data, strategy=params['role_focus'])

        optimal_squad, total_score = optimize_team(
            scored_data,
            budget=budget,
            must_include=must_include_players,
            must_exclude=must_exclude_players,
            retention_prices=params['retention_prices'],
            custom_auction_prices=params['custom_auction_prices'],
            constraint_mode=params['constraint_mode'],
            num_batters=params['num_batters'],
            num_bowlers=params['num_bowlers'],
            num_all_rounders=params['num_all_rounders'],
            num_pacers=params['num_pacers'],
            num_spinners=params['num_spinners'],
        )

        if optimal_squad is None or optimal_squad.empty:
            st.error("⚠️ Could not find a valid team! Budget may be too low, or constraints too tight. Try relaxing them.")
            return

        # Apply custom prices to the display copy
        for player, custom_price in params['retention_prices'].items():
            optimal_squad.loc[optimal_squad['Player Name'] == player, 'Cost_Cr'] = custom_price
        for player, custom_price in params['custom_auction_prices'].items():
            optimal_squad.loc[optimal_squad['Player Name'] == player, 'Cost_Cr'] = custom_price

        spent = optimal_squad['Cost_Cr'].sum()
        n_foreign = int(optimal_squad['Is_Foreign'].sum())

        # --- KPI Metrics ---
        st.subheader("📊 Team Analytics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Fantasy Points", f"{total_score:.2f}")
        m2.metric("Funds Spent", f"₹{spent:.2f} Cr")
        m3.metric("Funds Remaining", f"₹{budget - spent:.2f} Cr")
        m4.metric("Foreign Players", f"{n_foreign}/4")
        avg_value = optimal_squad['Value_Score'].mean()
        m5.metric("Avg Value Score", f"{avg_value:.3f}", help="Fantasy Score per ₹Cr — higher is more efficient.")

        st.markdown("---")

        # --- Charts ---
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            role_budget = optimal_squad.groupby('Role')['Cost_Cr'].sum().reset_index()
            fig_pie = px.pie(
                role_budget,
                values='Cost_Cr',
                names='Role',
                title="Budget Allocation by Role",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_chart2:
            fig_bar = px.bar(
                optimal_squad.sort_values('Fantasy_Score', ascending=True),
                x='Fantasy_Score',
                y='Player Name',
                color='Role',
                orientation='h',
                title="Player Fantasy Contributions",
                text_auto='.2f',
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_bar.update_layout(
                xaxis_title="Fantasy Score",
                yaxis_title="",
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # --- Value Score bar (the Moneyball insight in the team context) ---
        st.subheader("💎 Value Efficiency of Your Squad")
        st.caption("Value Score = Fantasy Score ÷ Auction Price. Higher bars = more value per rupee spent.")
        fig_value = px.bar(
            optimal_squad.sort_values('Value_Score', ascending=True),
            x='Value_Score',
            y='Player Name',
            color='Role',
            orientation='h',
            text_auto='.3f',
            title="Value Score by Player (Fantasy Points per ₹ Crore)",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_value.update_layout(
            xaxis_title="Value Score",
            yaxis_title="",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        st.plotly_chart(fig_value, use_container_width=True)

        st.markdown("---")

        # --- Final Squad Table ---
        st.subheader("📋 Your Optimal Squad")
        display_squad = optimal_squad.rename(columns={
            'Cost_Cr': 'Auction Price (Cr)',
            'Fantasy_Score': 'Fantasy Points',
            'Value_Score': 'Value Score',
            'Is_Foreign': 'Overseas Player',
        })
        st.dataframe(
            display_squad.style.background_gradient(subset=['Fantasy Points', 'Value Score'], cmap='YlGn'),
            use_container_width=True,
            hide_index=True,
            height=450,
        )

        csv = display_squad.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Squad as CSV",
            data=csv,
            file_name='optimal_ipl_squad.csv',
            mime='text/csv',
        )


def main():
    raw_data = load_data()
    player_list = sorted(raw_data['Player Name'].dropna().unique().tolist())

    params = render_sidebar(player_list, raw_data)
    if params is None:
        return

    st.title("🏆 IPL Moneyball: Data-Driven Team Optimization")
    st.caption(
        "Applying **Linear Programming** and **Value Analytics** to find the most efficient IPL auction squad — "
        "inspired by the Moneyball philosophy of finding undervalued players through statistics."
    )
    st.divider()

    # Pre-score data for Market Analysis and Scout tabs (independent of the optimizer run)
    scored_data = calculate_fantasy(raw_data, strategy=params.get('role_focus', 'Balanced'))

    tab_optimizer, tab_market, tab_scout = st.tabs([
        "⚙️ Team Optimizer",
        "📈 Market Analysis",
        "🔍 Player Scout",
    ])

    with tab_optimizer:
        render_optimizer_tab(params, raw_data)

    with tab_market:
        render_market_analysis_tab(scored_data)

    with tab_scout:
        render_player_scout_tab(scored_data)


if __name__ == "__main__":
    main()

    main()