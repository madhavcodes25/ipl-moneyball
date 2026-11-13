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


@st.cache_data
def get_scored_data(strategy: str):
    """Cache scored data per strategy so switching tabs doesn't re-run the scorer."""
    raw = load_data()
    return calculate_fantasy(raw, strategy=strategy)


def render_sidebar(player_list, raw_data):
    """Renders the Strategy Room sidebar and returns all user-configured parameters."""
    with st.sidebar:
        st.header("Strategy Room 🏏")
        st.markdown("Fine-tune your auction constraints.")

        constraint_mode = st.radio(
            "Select Constraint Mode",
            ["Flexible (Auto-balance)", "Strict (Custom Roles)"],
            key="sidebar_constraint_mode",
            help="Flexible ensures min 5 bowlers and 6 batters. Strict lets you pick exact counts.",
        )

        budget = st.slider(
            "Total Budget (Crores)", 20, 120, 100,
            key="sidebar_budget",
        )
        role_focus = st.selectbox(
            "Team Strategy", ["Balanced", "Batting Heavy", "Bowling Heavy"],
            key="sidebar_role_focus",
        )

        st.markdown("---")
        st.subheader("Team Composition")

        # Defaults used in Flexible mode (and as starting values in Strict mode)
        num_batters, num_bowlers, num_all_rounders = 5, 4, 2
        num_pacers, num_spinners = 3, 2

        if constraint_mode == "Strict (Custom Roles)":
            st.caption("Batters + Bowlers + All-Rounders must equal 11")
            col1, col2 = st.columns(2)
            with col1:
                num_batters = st.number_input(
                    "Batters 🏏", min_value=1, max_value=11, value=5,
                    key="sidebar_num_batters",
                )
            with col2:
                num_bowlers = st.number_input(
                    "Bowlers 🎯", min_value=0, max_value=11, value=4,
                    key="sidebar_num_bowlers",
                )

            num_all_rounders = st.number_input(
                "All-Rounders ⚔️", min_value=0, max_value=11, value=2,
                key="sidebar_num_all_rounders",
            )

            st.subheader("Bowling Attack Setup")
            col3, col4 = st.columns(2)
            with col3:
                num_pacers = st.number_input(
                    "Min Pacers", min_value=0, max_value=11, value=3,
                    key="sidebar_num_pacers",
                )
            with col4:
                num_spinners = st.number_input(
                    "Min Spinners", min_value=0, max_value=11, value=2,
                    key="sidebar_num_spinners",
                )

            # Only validate total in Strict mode — Flexible handles composition automatically
            if (num_batters + num_bowlers + num_all_rounders) != 11:
                st.error("🚨 Batters + Bowlers + All-Rounders must equal 11.")
                return None
        else:
            st.info(
                "🤖 **Auto-balance Active**\n\n"
                "The algorithm will automatically build the most optimal squad ensuring at least:\n"
                "- 6 Batting options\n"
                "- 5 Bowling options\n"
                "- 1 Wicket Keeper"
            )

        st.markdown("---")
        st.subheader("Retentions")
        must_include_players = st.multiselect(
            "Force include up to 11 players:",
            options=player_list,
            max_selections=11,
            key="sidebar_must_include",
            help="Will build the rest of the team around these retained players.",
        )

        retention_prices = {}
        if must_include_players:
            st.caption("Set custom retention prices (₹ Cr)")
            for player in must_include_players:
                default_cost = float(
                    raw_data[raw_data["Player Name"] == player]["Cost_Cr"].iloc[0]
                )
                retention_prices[player] = st.number_input(
                    f"{player} Price",
                    min_value=0.0,
                    max_value=float(budget),
                    value=default_cost,
                    step=0.25,
                    format="%.2f",
                    key=f"sidebar_retention_{player}",
                )

        st.markdown("---")
        st.subheader("Exclusions")
        must_exclude_players = st.multiselect(
            "Ignore these players:",
            options=player_list,
            key="sidebar_must_exclude",
            help="These players will be completely ignored by the optimizer.",
        )
        st.markdown("---")

        with st.expander("⚡ Phase Specialist Requirements"):
            st.caption(
                "Force the optimizer to include at least one specialist for a specific game phase. "
                "Only applies to players who have sufficient ball-by-ball data from IPL 2022–2026."
            )
            require_pp_bat = st.checkbox(
                "🏏 Require Powerplay Bat Specialist",
                value=False,
                key="sidebar_require_pp_bat",
                help="Ensures at least one player with an elite Powerplay (overs 1–6) strike rate is selected.",
            )
            require_death_bat = st.checkbox(
                "💥 Require Death Bat Specialist",
                value=False,
                key="sidebar_require_death_bat",
                help="Ensures at least one player with an elite Death Overs (16–20) strike rate is selected.",
            )
            require_pp_bowl = st.checkbox(
                "🎯 Require Powerplay Bowl Specialist",
                value=False,
                key="sidebar_require_pp_bowl",
                help="Ensures at least one bowler with an elite Powerplay economy is selected.",
            )
            require_death_bowl = st.checkbox(
                "🔥 Require Death Bowl Specialist",
                value=False,
                key="sidebar_require_death_bowl",
                help="Ensures at least one bowler with an elite Death Overs economy is selected.",
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
                key="sidebar_players_to_adjust",
                help="Override the dataset's default price for these players.",
            )
            custom_auction_prices = {}
            for player in players_to_adjust:
                default_cost = float(
                    raw_data[raw_data["Player Name"] == player]["Cost_Cr"].iloc[0]
                )
                custom_auction_prices[player] = st.number_input(
                    f"{player} Expected Price",
                    min_value=0.0,
                    max_value=float(budget),
                    value=default_cost,
                    step=0.25,
                    format="%.2f",
                    key=f"sidebar_customprice_{player}",
                )

        st.markdown("---")
        generate_btn = st.button(
            "Draft Perfect Team", type="primary", use_container_width=True,
            key="sidebar_generate_btn",
        )

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
        require_pp_bat=require_pp_bat,
        require_death_bat=require_death_bat,
        require_pp_bowl=require_pp_bowl,
        require_death_bowl=require_death_bowl,
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

    display_df = scored_data.dropna(subset=["Cost_Cr", "Fantasy_Score"]).copy()
    display_df = display_df[display_df["Cost_Cr"] > 0]

    frontier_df = get_efficient_frontier(display_df)

    # Scatter plot
    fig = px.scatter(
        display_df,
        x="Cost_Cr",
        y="Fantasy_Score",
        color="Role",
        size="Value_Score",
        hover_name="Player Name",
        hover_data={
            "Cost_Cr": ":.2f",
            "Fantasy_Score": ":.3f",
            "Value_Score": ":.3f",
            "Role": True,
            "Subrole": True,
        },
        labels={
            "Cost_Cr": "Auction Price (₹ Cr)",
            "Fantasy_Score": "Fantasy Score",
            "Value_Score": "Value Score (Fantasy/₹Cr)",
        },
        title="Player Market Map: Price vs. Fantasy Performance",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.75,
        size_max=25,
    )

    if not frontier_df.empty:
        fig.add_trace(go.Scatter(
            x=frontier_df["Cost_Cr"],
            y=frontier_df["Fantasy_Score"],
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(color="gold", width=2.5, dash="dash"),
            marker=dict(size=8, symbol="diamond", color="gold", line=dict(color="black", width=1)),
        ))

    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(14,17,23,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True, key="market_scatter")

    st.markdown("---")

    col_gems, col_over = st.columns(2)

    with col_gems:
        st.subheader("💎 Hidden Gems (Most Undervalued)")
        st.caption("High fantasy output per crore spent — ideal auction targets.")
        gems = (
            display_df[display_df["Fantasy_Score"] > display_df["Fantasy_Score"].median()]
            .sort_values("Value_Score", ascending=False)
            .head(10)[["Player Name", "Role", "Cost_Cr", "Fantasy_Score", "Value_Score"]]
            .reset_index(drop=True)
        )
        gems.index += 1
        gems.columns = ["Player", "Role", "Price (Cr)", "Fantasy Score", "Value Score"]
        gems["Price (Cr)"] = gems["Price (Cr)"].map("₹{:.2f} Cr".format)
        gems["Fantasy Score"] = gems["Fantasy Score"].map("{:.2f}".format)
        gems["Value Score"] = gems["Value Score"].map("{:.3f}".format)
        st.dataframe(gems, use_container_width=True, height=380)

    with col_over:
        st.subheader("💸 Overpaid Players (Worst Value)")
        st.caption("High price tag but low fantasy returns — players to avoid at auction.")
        overpaid = (
            display_df[display_df["Cost_Cr"] >= display_df["Cost_Cr"].median()]
            .sort_values("Value_Score", ascending=True)
            .head(10)[["Player Name", "Role", "Cost_Cr", "Fantasy_Score", "Value_Score"]]
            .reset_index(drop=True)
        )
        overpaid.index += 1
        overpaid.columns = ["Player", "Role", "Price (Cr)", "Fantasy Score", "Value Score"]
        overpaid["Price (Cr)"] = overpaid["Price (Cr)"].map("₹{:.2f} Cr".format)
        overpaid["Fantasy Score"] = overpaid["Fantasy Score"].map("{:.2f}".format)
        overpaid["Value Score"] = overpaid["Value Score"].map("{:.3f}".format)
        st.dataframe(overpaid, use_container_width=True, height=380)

    st.markdown("---")

    st.subheader("🎯 Value Score Distribution by Role")
    st.caption("Box plot showing the spread of Value Scores per role — useful for spotting which positions have the most inefficiency.")
    fig_box = px.box(
        display_df,
        x="Role",
        y="Value_Score",
        color="Role",
        points="all",
        hover_name="Player Name",
        labels={"Value_Score": "Value Score (Fantasy/₹Cr)", "Role": "Player Role"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_box.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(14,17,23,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig_box, use_container_width=True, key="market_box")


def render_player_scout_tab(scored_data: pd.DataFrame):
    """Renders an interactive player comparison and scouting table."""
    st.subheader("🔍 Player Scouting Database")
    st.caption("Search, filter, and sort all players by any metric. Great for live auction preparation.")

    col1, col2, col3 = st.columns(3)
    with col1:
        role_filter = st.multiselect(
            "Filter by Role",
            options=sorted(scored_data["Role"].dropna().unique()),
            default=[],
            key="scout_role_filter",
        )
    with col2:
        nationality_filter = st.radio(
            "Nationality", ["All", "Indian", "Overseas"],
            horizontal=True,
            key="scout_nationality_filter",
        )
    with col3:
        max_score = float(scored_data["Fantasy_Score"].max())
        min_score = st.slider(
            "Min Fantasy Score",
            min_value=0.0,
            max_value=max_score if max_score > 0 else 1.0,
            value=0.0,
            step=0.1,
            key="scout_min_score",
        )

    scout_df = scored_data.copy()
    if role_filter:
        scout_df = scout_df[scout_df["Role"].isin(role_filter)]
    if nationality_filter == "Indian":
        scout_df = scout_df[scout_df["Is_Foreign"] == 0]
    elif nationality_filter == "Overseas":
        scout_df = scout_df[scout_df["Is_Foreign"] == 1]
    scout_df = scout_df[scout_df["Fantasy_Score"] >= min_score]

    display_cols = [
        "Player Name", "Role", "Subrole", "Cost_Cr", "Is_Foreign",
        "Batting Runs", "Batting Average", "Batting Strike Rate",
        "Wickets", "Economy Rate", "Fantasy_Score", "Value_Score",
    ]
    # Only keep columns that actually exist in the dataframe
    display_cols = [c for c in display_cols if c in scout_df.columns]

    scout_display = (
        scout_df[display_cols]
        .sort_values("Value_Score", ascending=False)
        .reset_index(drop=True)
    )
    scout_display.index += 1

    scout_display = scout_display.rename(columns={
        "Cost_Cr": "Price (Cr)",
        "Is_Foreign": "Overseas",
        "Fantasy_Score": "Fantasy Score",
        "Value_Score": "Value Score",
        "Batting Runs": "Runs",
        "Batting Average": "Bat Avg",
        "Batting Strike Rate": "SR",
        "Economy Rate": "Econ",
    })

    # Format numbers cleanly so they don't display like 16.000000
    if "Price (Cr)" in scout_display.columns:
        scout_display["Price (Cr)"] = scout_display["Price (Cr)"].round(2)
    if "Fantasy Score" in scout_display.columns:
        scout_display["Fantasy Score"] = scout_display["Fantasy Score"].round(2)
    if "Value Score" in scout_display.columns:
        scout_display["Value Score"] = scout_display["Value Score"].round(3)
    if "Bat Avg" in scout_display.columns:
        scout_display["Bat Avg"] = scout_display["Bat Avg"].round(2)
    if "SR" in scout_display.columns:
        scout_display["SR"] = scout_display["SR"].round(1)
    if "Econ" in scout_display.columns:
        scout_display["Econ"] = scout_display["Econ"].round(2)
    if "Runs" in scout_display.columns:
        scout_display["Runs"] = scout_display["Runs"].fillna(0).astype(int)
    if "Wickets" in scout_display.columns:
        scout_display["Wickets"] = scout_display["Wickets"].fillna(0).astype(int)

    scout_column_config = {
        "Price (Cr)": st.column_config.NumberColumn("Price (Cr)", format="₹%.2f Cr"),
        "Fantasy Score": st.column_config.NumberColumn("Fantasy Score", format="%.2f"),
        "Value Score": st.column_config.NumberColumn("Value Score", format="%.3f"),
        "Bat Avg": st.column_config.NumberColumn("Bat Avg", format="%.2f"),
        "SR": st.column_config.NumberColumn("SR", format="%.1f"),
        "Econ": st.column_config.NumberColumn("Econ", format="%.2f"),
        "Runs": st.column_config.NumberColumn("Runs", format="%d"),
        "Wickets": st.column_config.NumberColumn("Wickets", format="%d"),
    }

    # Use background_gradient only on numeric columns that definitely exist
    gradient_cols = [c for c in ["Fantasy Score", "Value Score"] if c in scout_display.columns]
    styled_scout = (
        scout_display.style.background_gradient(subset=gradient_cols, cmap="YlGn")
        if gradient_cols else scout_display
    )
    st.dataframe(
        styled_scout,
        use_container_width=True,
        height=500,
        column_config=scout_column_config,
    )

    st.caption(f"Showing {len(scout_display)} players. Sorted by Value Score (descending).")


def render_optimizer_tab(params: dict, scored_data: pd.DataFrame, raw_data: pd.DataFrame):
    """Runs the LP optimizer and renders team results."""
    if not params["generate_btn"]:
        st.info("👈 Adjust your parameters in the **Strategy Room** and click **'Draft Perfect Team'** to begin.")
        return

    must_include_players = params["must_include_players"]
    must_exclude_players = params["must_exclude_players"]
    budget = params["budget"]

    # Input validation
    conflict_players = set(must_include_players).intersection(set(must_exclude_players))
    if conflict_players:
        st.error(f"🚨 Logic Error: You cannot both retain and ignore: {', '.join(conflict_players)}")
        return

    if (params["num_pacers"] + params["num_spinners"]) > (params["num_bowlers"] + params["num_all_rounders"]):
        st.warning("⚠️ Logic Warning: More Pacers + Spinners requested than total bowling slots. Optimizer may fail.")

    if must_include_players:
        retained_df = raw_data[raw_data["Player Name"].isin(must_include_players)]
        budget_spent_retained = sum(params["retention_prices"].values())
        overseas_count_retained = retained_df["Is_Foreign"].sum() if "Is_Foreign" in retained_df.columns else 0

        if overseas_count_retained > 4:
            st.error(f"🚨 Rule Violation: {int(overseas_count_retained)} retained overseas players — max is 4.")
            return
        if budget_spent_retained > budget:
            st.error(f"🚨 Budget Exceeded: Retained players cost ₹{budget_spent_retained:.2f} Cr — over your ₹{budget} Cr limit.")
            return

    with st.spinner("Crunching the numbers and simulating auction bids..."):
        optimal_squad, total_score = optimize_team(
            scored_data,
            budget=budget,
            must_include=must_include_players,
            must_exclude=must_exclude_players,
            retention_prices=params["retention_prices"],
            custom_auction_prices=params["custom_auction_prices"],
            constraint_mode=params["constraint_mode"],
            num_batters=params["num_batters"],
            num_bowlers=params["num_bowlers"],
            num_all_rounders=params["num_all_rounders"],
            num_pacers=params["num_pacers"],
            num_spinners=params["num_spinners"],
            require_pp_bat_specialist=params.get("require_pp_bat", False),
            require_death_bat_specialist=params.get("require_death_bat", False),
            require_pp_bowl_specialist=params.get("require_pp_bowl", False),
            require_death_bowl_specialist=params.get("require_death_bowl", False),
        )

    if optimal_squad is None or optimal_squad.empty:
        st.error("⚠️ Could not find a valid team! Budget may be too low, or constraints too tight. Try relaxing them.")
        return

    # Apply custom prices to the display copy
    for player, custom_price in params["retention_prices"].items():
        optimal_squad.loc[optimal_squad["Player Name"] == player, "Cost_Cr"] = custom_price
    for player, custom_price in params["custom_auction_prices"].items():
        optimal_squad.loc[optimal_squad["Player Name"] == player, "Cost_Cr"] = custom_price

    spent = optimal_squad["Cost_Cr"].sum()
    n_foreign = int(optimal_squad["Is_Foreign"].sum())
    avg_value = optimal_squad["Value_Score"].mean()

    # KPI Metrics
    st.subheader("📊 Team Analytics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Fantasy Points", f"{total_score:.2f}")
    m2.metric("Funds Spent", f"₹{spent:.2f} Cr")
    m3.metric("Funds Remaining", f"₹{budget - spent:.2f} Cr")
    m4.metric("Foreign Players", f"{n_foreign}/4")
    m5.metric("Avg Value Score", f"{avg_value:.3f}", help="Fantasy Score per ₹Cr — higher is more efficient.")

    # Phase Specialist badges — show which constraints are active
    active_badges = []
    if params.get("require_pp_bat"):    active_badges.append("🏏 PP Bat Specialist")
    if params.get("require_death_bat"): active_badges.append("💥 Death Bat Specialist")
    if params.get("require_pp_bowl"):   active_badges.append("🎯 PP Bowl Specialist")
    if params.get("require_death_bowl"):active_badges.append("🔥 Death Bowl Specialist")
    if active_badges:
        st.info(f"**Phase Constraints Active:** {' · '.join(active_badges)} — the optimizer guaranteed these roles are covered.")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        role_budget = optimal_squad.groupby("Role")["Cost_Cr"].sum().reset_index()
        fig_pie = px.pie(
            role_budget,
            values="Cost_Cr",
            names="Role",
            title="Budget Allocation by Role",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_pie, use_container_width=True, key="optimizer_pie")

    with col_chart2:
        fig_bar = px.bar(
            optimal_squad.sort_values("Fantasy_Score", ascending=True),
            x="Fantasy_Score",
            y="Player Name",
            color="Role",
            orientation="h",
            title="Player Fantasy Contributions",
            text_auto=".2f",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_bar.update_layout(
            xaxis_title="Fantasy Score",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="optimizer_bar")

    st.markdown("---")

    st.subheader("💎 Value Efficiency of Your Squad")
    st.caption("Value Score = Fantasy Score ÷ Auction Price. Higher bars = more value per rupee spent.")
    fig_value = px.bar(
        optimal_squad.sort_values("Value_Score", ascending=True),
        x="Value_Score",
        y="Player Name",
        color="Role",
        orientation="h",
        text_auto=".3f",
        title="Value Score by Player (Fantasy Points per ₹ Crore)",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_value.update_layout(
        xaxis_title="Value Score",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig_value, use_container_width=True, key="optimizer_value_bar")

    st.markdown("---")

    st.subheader("📋 Your Optimal Squad")
    display_squad = optimal_squad.rename(columns={
        "Cost_Cr": "Auction Price (Cr)",
        "Fantasy_Score": "Fantasy Points",
        "Value_Score": "Value Score",
        "Is_Foreign": "Overseas Player",
    }).copy()

    # Build a "Phase Specialist" badge column from boolean tag columns
    specialist_tag_map = {
        "pp_bat_specialist":    "🏏PP",
        "death_bat_specialist": "💥Death Bat",
        "pp_bowl_specialist":   "🎯PP Bowl",
        "death_bowl_specialist":"🔥Death Bowl",
    }
    badges = []
    for _, row in display_squad.iterrows():
        player_badges = [
            label for col, label in specialist_tag_map.items()
            if col in display_squad.columns and row.get(col) is True
        ]
        badges.append(", ".join(player_badges) if player_badges else "—")
    display_squad["Phase Tags"] = badges

    # Drop the raw boolean columns — replaced by the badge column
    display_squad = display_squad.drop(
        columns=[c for c in specialist_tag_map if c in display_squad.columns]
    )

    # Format numeric columns cleanly so they don't display as 16.000000
    display_squad["Auction Price (Cr)"] = display_squad["Auction Price (Cr)"].round(2)
    display_squad["Fantasy Points"] = display_squad["Fantasy Points"].round(2)
    display_squad["Value Score"] = display_squad["Value Score"].round(3)
    if "Phase_Score" in display_squad.columns:
        display_squad["Phase_Score"] = display_squad["Phase_Score"].round(3)

    squad_column_config = {
        "Auction Price (Cr)": st.column_config.NumberColumn(
            "Auction Price (Cr)",
            format="₹%.2f Cr",
        ),
        "Fantasy Points": st.column_config.NumberColumn(
            "Fantasy Points",
            format="%.2f",
        ),
        "Value Score": st.column_config.NumberColumn(
            "Value Score",
            format="%.3f",
        ),
    }

    gradient_cols = [c for c in ["Fantasy Points", "Value Score"] if c in display_squad.columns]
    styled_squad = (
        display_squad.style.background_gradient(subset=gradient_cols, cmap="YlGn")
        if gradient_cols else display_squad
    )
    st.dataframe(
        styled_squad,
        use_container_width=True,
        hide_index=True,
        height=450,
        column_config=squad_column_config,
    )

    csv = display_squad.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Squad as CSV",
        data=csv,
        file_name="optimal_ipl_squad.csv",
        mime="text/csv",
        key="optimizer_download_btn",
    )


def render_phase_analysis_tab(scored_data: pd.DataFrame):
    """
    Renders the Phase Analysis tab showing Powerplay / Middle / Death specialist stats.

    Batting specialists ranked by composite: 0.6 × SR + 0.4 × Boundary%.
    Bowling specialists ranked by composite: 0.5 × (inv Economy) + 0.5 × Wickets/Over.
    """
    st.subheader("⚡ Phase Specialist Analysis")
    st.caption(
        "Ball-by-ball stats from **IPL 2022–2026** (RR Hackathon data) with recency-weighted averages. "
        "**Bat score** = SR + Boundary%  ·  **Bowl score** = Economy + Wickets/Over."
    )

    BAT_SR_COLS    = {"Powerplay": "pp_bat_sr",          "Middle": "middle_bat_sr",         "Death": "death_bat_sr"}
    BAT_BPCT_COLS  = {"Powerplay": "pp_bat_boundary_pct","Middle": "middle_bat_boundary_pct","Death": "death_bat_boundary_pct"}
    BOWL_EC_COLS   = {"Powerplay": "pp_bowl_econ",        "Middle": "middle_bowl_econ",       "Death": "death_bowl_econ"}
    BOWL_WPO_COLS  = {"Powerplay": "pp_bowl_wkts_po",     "Middle": "middle_bowl_wkts_po",    "Death": "death_bowl_wkts_po"}
    BOWL_WK_COLS   = {"Powerplay": "pp_bowl_wkts",        "Middle": "middle_bowl_wkts",       "Death": "death_bowl_wkts"}

    phase_available = any(col in scored_data.columns for col in BAT_SR_COLS.values())
    if not phase_available:
        st.warning("Phase data not available. Ensure the RR Hackathon data is present in `data/RR-Hackathon-Data/`.")
        return

    all_phase_cols = (
        list(BAT_SR_COLS.values()) + list(BAT_BPCT_COLS.values()) +
        list(BOWL_EC_COLS.values()) + list(BOWL_WPO_COLS.values()) + list(BOWL_WK_COLS.values())
    )
    phase_df = scored_data[
        ["Player Name", "Role", "Cost_Cr"] + [c for c in all_phase_cols if c in scored_data.columns]
    ].copy()

    phase_select = st.radio(
        "Select Phase",
        ["Powerplay (Overs 1–6)", "Middle Overs (7–15)", "Death Overs (16–20)"],
        horizontal=True,
        key="phase_analysis_select",
    )
    phase_key = "Powerplay" if "Powerplay" in phase_select else ("Middle" if "Middle" in phase_select else "Death")

    st.divider()
    col_bat, col_bowl = st.columns(2)

    # ---- Batting Specialists: SR + Boundary% composite ----
    with col_bat:
        bat_col  = BAT_SR_COLS[phase_key]
        bpct_col = BAT_BPCT_COLS[phase_key]
        st.markdown(f"### 🏏 {phase_select} Batting Specialists")
        st.caption("Score = **0.6 × SR** + **0.4 × Boundary%** — rewards aggression AND boundary-hitting ability.")
        if bat_col in phase_df.columns:
            bat_view = phase_df[phase_df[bat_col].notna()].copy()
            bat_view = bat_view.rename(columns={bat_col: "Strike Rate"})
            if bpct_col in bat_view.columns:
                bat_view = bat_view.rename(columns={bpct_col: "Boundary%"})

            sr_max = bat_view["Strike Rate"].max(); sr_min = bat_view["Strike Rate"].min()
            bat_view["_norm_sr"] = ((bat_view["Strike Rate"] - sr_min) / (sr_max - sr_min + 1e-6)).clip(0, 1)
            if "Boundary%" in bat_view.columns:
                bp_max = bat_view["Boundary%"].max(); bp_min = bat_view["Boundary%"].min()
                bat_view["_norm_bp"] = ((bat_view["Boundary%"] - bp_min) / (bp_max - bp_min + 1e-6)).clip(0, 1)
                bat_view["Composite"] = (0.6 * bat_view["_norm_sr"] + 0.4 * bat_view["_norm_bp"]).round(3)
            else:
                bat_view["Composite"] = bat_view["_norm_sr"].round(3)

            drop_cols = [c for c in ["_norm_sr", "_norm_bp"] if c in bat_view.columns]
            keep_cols = ["Player Name", "Role", "Strike Rate"] + (["Boundary%"] if "Boundary%" in bat_view.columns else []) + ["Composite"]
            bat_top = bat_view[keep_cols].drop(columns=drop_cols, errors="ignore").sort_values("Composite", ascending=False).head(15).reset_index(drop=True)
            bat_top.index += 1

            if not bat_top.empty:
                fig_bat = px.bar(
                    bat_top.head(10), x="Player Name", y="Composite", color="Role",
                    title=f"Top 10 — {phase_select} Batting Composite",
                    text_auto=".3f", color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_bat.update_layout(xaxis_tickangle=-35, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig_bat, use_container_width=True, key=f"phase_bat_bar_{phase_key}")
                bat_top["Strike Rate"] = bat_top["Strike Rate"].round(1)
                if "Boundary%" in bat_top.columns:
                    bat_top["Boundary%"] = bat_top["Boundary%"].round(1)
                col_cfg_bat = {
                    "Strike Rate": st.column_config.NumberColumn("SR",         format="%.1f"),
                    "Boundary%":   st.column_config.NumberColumn("Boundary%",  format="%.1f"),
                    "Composite":   st.column_config.NumberColumn("Score",       format="%.3f"),
                }
                st.dataframe(bat_top, use_container_width=True, height=350, hide_index=False, column_config=col_cfg_bat)
            else:
                st.info("No batting data available for this phase.")
        else:
            st.info("Batting phase data not loaded.")

    # ---- Bowling Specialists: Economy + Wkts/Over composite ----
    with col_bowl:
        bowl_ec_col  = BOWL_EC_COLS[phase_key]
        bowl_wpo_col = BOWL_WPO_COLS[phase_key]
        bowl_wk_col  = BOWL_WK_COLS[phase_key]
        st.markdown(f"### 🎯 {phase_select} Bowling Specialists")
        st.caption("Score = **0.5 × Economy** (inverted) + **0.5 × Wkts/Over** — rewards containment AND wicket-taking threat.")
        if bowl_ec_col in phase_df.columns:
            bowl_view = phase_df[phase_df[bowl_ec_col].notna()].copy()
            bowl_view = bowl_view.rename(columns={bowl_ec_col: "Economy"})
            if bowl_wpo_col in bowl_view.columns:
                bowl_view = bowl_view.rename(columns={bowl_wpo_col: "Wkts/Over"})
            if bowl_wk_col in bowl_view.columns:
                bowl_view = bowl_view.rename(columns={bowl_wk_col: "Wkts (Total)"})

            ec_max = bowl_view["Economy"].max(); ec_min = bowl_view["Economy"].min()
            bowl_view["_norm_econ"] = ((ec_max - bowl_view["Economy"]) / (ec_max - ec_min + 1e-6)).clip(0, 1)
            if "Wkts/Over" in bowl_view.columns:
                wp_max = bowl_view["Wkts/Over"].max(); wp_min = bowl_view["Wkts/Over"].min()
                bowl_view["_norm_wpo"] = ((bowl_view["Wkts/Over"] - wp_min) / (wp_max - wp_min + 1e-6)).clip(0, 1)
                bowl_view["Composite"] = (0.5 * bowl_view["_norm_econ"] + 0.5 * bowl_view["_norm_wpo"]).round(3)
            else:
                bowl_view["Composite"] = bowl_view["_norm_econ"].round(3)

            keep_cols = ["Player Name", "Role", "Economy"] + (["Wkts/Over"] if "Wkts/Over" in bowl_view.columns else []) + (["Wkts (Total)"] if "Wkts (Total)" in bowl_view.columns else []) + ["Composite"]
            norm_drop = [c for c in ["_norm_econ", "_norm_wpo"] if c in bowl_view.columns]
            bowl_top = bowl_view[[c for c in keep_cols if c in bowl_view.columns]].sort_values("Composite", ascending=False).head(15).reset_index(drop=True)
            bowl_top.index += 1

            if not bowl_top.empty:
                fig_bowl = px.bar(
                    bowl_top.head(10), x="Player Name", y="Composite", color="Role",
                    title=f"Top 10 — {phase_select} Bowling Composite",
                    text_auto=".3f", color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_bowl.update_layout(xaxis_tickangle=-35, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig_bowl, use_container_width=True, key=f"phase_bowl_bar_{phase_key}")
                bowl_top["Economy"] = bowl_top["Economy"].round(2)
                col_cfg_bowl = {
                    "Economy":     st.column_config.NumberColumn("Economy",    format="%.2f"),
                    "Composite":   st.column_config.NumberColumn("Score",       format="%.3f"),
                }
                if "Wkts/Over" in bowl_top.columns:
                    bowl_top["Wkts/Over"] = bowl_top["Wkts/Over"].round(3)
                    col_cfg_bowl["Wkts/Over"] = st.column_config.NumberColumn("Wkts/Over", format="%.3f")
                if "Wkts (Total)" in bowl_top.columns:
                    bowl_top["Wkts (Total)"] = bowl_top["Wkts (Total)"].fillna(0).astype(int)
                    col_cfg_bowl["Wkts (Total)"] = st.column_config.NumberColumn("Wkts", format="%d")
                st.dataframe(bowl_top, use_container_width=True, height=350, hide_index=False, column_config=col_cfg_bowl)
            else:
                st.info("No bowling data available for this phase.")
        else:
            st.info("Bowling phase data not loaded.")

    # ----------------------------------------------------------------
    # Phase Heatmap
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### 🌡️ Phase Specialist Heatmap")
    st.caption("Multi-phase comparison for top players. Shows batting SR and bowling economy across all three phases.")

    heatmap_tab_bat, heatmap_tab_bowl = st.tabs(["Batting SR", "Bowling Economy"])

    with heatmap_tab_bat:
        bat_heat_cols = {k: v for k, v in {"Powerplay": "pp_bat_sr", "Middle": "middle_bat_sr", "Death": "death_bat_sr"}.items() if v in scored_data.columns}
        if bat_heat_cols:
            heat_df = scored_data[["Player Name", "Fantasy_Score"] + list(bat_heat_cols.values())].copy()
            heat_df = heat_df.dropna(subset=list(bat_heat_cols.values()), how="all")
            heat_df = heat_df.nlargest(30, "Fantasy_Score")
            heat_matrix = heat_df.set_index("Player Name")[list(bat_heat_cols.values())]
            heat_matrix.columns = list(bat_heat_cols.keys())
            fig_heat_bat = px.imshow(
                heat_matrix, color_continuous_scale="YlOrRd",
                title="Batting Strike Rate by Phase (Top 30 Players by Fantasy Score)",
                labels=dict(color="Strike Rate"), aspect="auto", text_auto=".0f",
            )
            fig_heat_bat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_heat_bat, use_container_width=True, key="phase_heatmap_bat")
        else:
            st.info("Batting heatmap data not available.")

    with heatmap_tab_bowl:
        bowl_heat_cols = {k: v for k, v in {"Powerplay": "pp_bowl_econ", "Middle": "middle_bowl_econ", "Death": "death_bowl_econ"}.items() if v in scored_data.columns}
        if bowl_heat_cols:
            heat_df = scored_data[["Player Name", "Fantasy_Score"] + list(bowl_heat_cols.values())].copy()
            heat_df = heat_df.dropna(subset=list(bowl_heat_cols.values()), how="all")
            heat_df = heat_df.nlargest(30, "Fantasy_Score")
            heat_matrix = heat_df.set_index("Player Name")[list(bowl_heat_cols.values())]
            heat_matrix.columns = list(bowl_heat_cols.keys())
            fig_heat_bowl = px.imshow(
                heat_matrix, color_continuous_scale="Blues_r",
                title="Bowling Economy by Phase — lower = better (Top 30 Players)",
                labels=dict(color="Economy"), aspect="auto", text_auto=".1f",
            )
            fig_heat_bowl.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_heat_bowl, use_container_width=True, key="phase_heatmap_bowl")
        else:
            st.info("Bowling heatmap data not available.")

    # ----------------------------------------------------------------
    # All-Phase Summary Table
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### 📊 All-Phase Specialist Summary")
    summary_base_cols = ["Player Name", "Role", "Cost_Cr"]
    phase_stat_candidates = [
        "pp_bat_sr", "middle_bat_sr", "death_bat_sr",
        "pp_bat_boundary_pct", "middle_bat_boundary_pct", "death_bat_boundary_pct",
        "pp_bowl_econ", "middle_bowl_econ", "death_bowl_econ",
        "pp_bowl_wkts_po", "middle_bowl_wkts_po", "death_bowl_wkts_po",
    ]
    present_phase_cols = [c for c in phase_stat_candidates if c in scored_data.columns]
    summary_df = scored_data[summary_base_cols + present_phase_cols].copy()
    summary_df = summary_df.dropna(subset=present_phase_cols, how="all").copy()

    rename_map = {
        "Cost_Cr": "Price (Cr)",
        "pp_bat_sr": "PP SR", "middle_bat_sr": "Mid SR", "death_bat_sr": "Death SR",
        "pp_bat_boundary_pct": "PP Bdry%", "middle_bat_boundary_pct": "Mid Bdry%", "death_bat_boundary_pct": "Death Bdry%",
        "pp_bowl_econ": "PP Econ", "middle_bowl_econ": "Mid Econ", "death_bowl_econ": "Death Econ",
        "pp_bowl_wkts_po": "PP W/Over", "middle_bowl_wkts_po": "Mid W/Over", "death_bowl_wkts_po": "Death W/Over",
    }
    summary_df = summary_df.rename(columns={k: v for k, v in rename_map.items() if k in summary_df.columns})
    sort_col = "PP SR" if "PP SR" in summary_df.columns else summary_df.columns[2]
    summary_df = summary_df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    summary_df.index += 1

    for col in summary_df.select_dtypes(include="number").columns:
        summary_df[col] = summary_df[col].round(2)

    summary_col_cfg = {
        "Price (Cr)":   st.column_config.NumberColumn("Price (Cr)", format="₹%.2f Cr"),
        "PP SR":        st.column_config.NumberColumn("PP SR",       format="%.1f"),
        "Mid SR":       st.column_config.NumberColumn("Mid SR",      format="%.1f"),
        "Death SR":     st.column_config.NumberColumn("Death SR",    format="%.1f"),
        "PP Bdry%":     st.column_config.NumberColumn("PP Bdry%",    format="%.1f"),
        "Mid Bdry%":    st.column_config.NumberColumn("Mid Bdry%",   format="%.1f"),
        "Death Bdry%":  st.column_config.NumberColumn("Death Bdry%", format="%.1f"),
        "PP Econ":      st.column_config.NumberColumn("PP Econ",     format="%.2f"),
        "Mid Econ":     st.column_config.NumberColumn("Mid Econ",    format="%.2f"),
        "Death Econ":   st.column_config.NumberColumn("Death Econ",  format="%.2f"),
        "PP W/Over":    st.column_config.NumberColumn("PP W/Over",   format="%.3f"),
        "Mid W/Over":   st.column_config.NumberColumn("Mid W/Over",  format="%.3f"),
        "Death W/Over": st.column_config.NumberColumn("Death W/Over",format="%.3f"),
    }
    st.dataframe(summary_df, use_container_width=True, height=400, column_config=summary_col_cfg)
    st.caption(f"Showing {len(summary_df)} players with phase data. Sorted by Powerplay SR descending.")


def main():
    raw_data = load_data()
    player_list = sorted(raw_data["Player Name"].dropna().unique().tolist())

    params = render_sidebar(player_list, raw_data)
    if params is None:
        return

    st.title("🏆 IPL Moneyball: Data-Driven Team Optimization")
    st.caption(
        "Applying **Linear Programming** and **Value Analytics** to find the most efficient IPL auction squad — "
        "inspired by the Moneyball philosophy of finding undervalued players through statistics."
    )
    st.divider()

    scored_data = get_scored_data(params["role_focus"])

    tab_optimizer, tab_market, tab_scout, tab_phase = st.tabs([
        "⚙️ Team Optimizer",
        "📈 Market Analysis",
        "🔍 Player Scout",
        "⚡ Phase Analysis",
    ])

    with tab_optimizer:
        render_optimizer_tab(params, scored_data, raw_data)

    with tab_market:
        render_market_analysis_tab(scored_data)

    with tab_scout:
        render_player_scout_tab(scored_data)

    with tab_phase:
        render_phase_analysis_tab(scored_data)


if __name__ == "__main__":
    main()