import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_cleaning import prepare_data
from src.optimizer import calculate_fantasy, optimize_team

st.set_page_config(page_title="IPL Moneyball", page_icon="🏏", layout="wide")

@st.cache_data
def load_data():
    return prepare_data()

def main():
    raw_data = load_data()
    player_list = sorted(raw_data['Player Name'].dropna().unique().tolist())

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
                    st.error("🚨 Constraint Error: The sum of Batters, Bowlers, All-Rounders, and Wicket Keepers must exactly equal 11.")
                    return
            
            
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
                    # Fetch the player's default cost from the dataset to use as a starting value
                    default_cost = float(raw_data[raw_data['Player Name'] == player]['Cost_Cr'].iloc[0])
                    
                    # Create a number input for each retained player
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
        st.markdown("---")
        generate_btn = st.button("Draft Perfect Team", type="primary", use_container_width=True)

    st.title("🏆 Data-Driven IPL Team Selector")
    st.caption(f"Building the best mathematically possible team using the '{role_focus}' strategy.")
    st.divider()

    if not generate_btn:
        st.info("👈 Adjust your parameters in the Strategy Room and click 'Draft Perfect Team' to begin.")
        return
    
    conflict_players = set(must_include_players).intersection(set(must_exclude_players))
    if conflict_players:
        st.error(f"🚨 Logic Error: You cannot both retain and ignore the same player(s): {', '.join(conflict_players)}")
        return
    
    if (num_batters + num_bowlers + num_all_rounders) != 11:
        st.error("🚨 Constraint Error: The sum of Batters, Bowlers, All-Rounders, and Wicket Keepers must exactly equal 11.")
        return
        
    if (num_pacers + num_spinners) > (num_bowlers + num_all_rounders):
        st.warning("⚠️ Logic Warning: You have requested more Pacers and Spinners than total available bowling options (Bowlers + All-Rounders). The optimizer may fail.")

    if must_include_players:
        retained_df = raw_data[raw_data['Player Name'].isin(must_include_players)]
        
        cost_col = 'Cost_Cr' if 'Cost_Cr' in retained_df.columns else 'Price'
        foreign_col = 'Is_Foreign' if 'Is_Foreign' in retained_df.columns else 'Overseas'
        
        budget_spent_retained = sum(retention_prices.values())
        overseas_count_retained = retained_df[foreign_col].sum() if foreign_col in retained_df.columns else 0
        
        if overseas_count_retained > 4:
            st.error(f"🚨 Rule Violation: You have retained {int(overseas_count_retained)} overseas players. The maximum allowed is 4.")
            return 
            
        if budget_spent_retained > budget:
            st.error(f"🚨 Budget Exceeded: Your retained players cost ₹{budget_spent_retained:.2f} Cr, which is over your ₹{budget} Cr limit.")
            return 
            


    with st.spinner("Crunching the numbers and simulating auction bids..."):
        
        scored_data = calculate_fantasy(raw_data, strategy=role_focus)
        
        optimal_squad, total_score = optimize_team(
            scored_data, 
            budget=budget,
            must_include=must_include_players,
            must_exclude=must_exclude_players,
            retention_prices=retention_prices,
            custom_auction_prices=custom_auction_prices,
            num_batters=num_batters,
            constraint_mode=constraint_mode,
            num_bowlers=num_bowlers,
            num_all_rounders=num_all_rounders,
            num_pacers=num_pacers,
            num_spinners=num_spinners
        )
        
        if optimal_squad is not None and not optimal_squad.empty:

            for player, custom_price in retention_prices.items():
                optimal_squad.loc[optimal_squad['Player Name'] == player, 'Cost_Cr'] = custom_price

            for player, custom_price in custom_auction_prices.items(): 
                optimal_squad.loc[optimal_squad['Player Name'] == player, 'Cost_Cr'] = custom_price

            spent = optimal_squad['Cost_Cr'].sum()
            
            display_squad = optimal_squad.rename(columns={
                'Cost_Cr': 'Auction Price (Cr)',
                'Fantasy_Score': 'Fantasy Points',
                'Is_Foreign': 'Overseas Player'
            })

            st.subheader("📊 Team Analytics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(label="Total Fantasy Points", value=f"{total_score:.2f}") 
            m2.metric(label="Funds Spent", value=f"₹{spent:.2f} Cr")
            m3.metric(label="Funds Remaining", value=f"₹{budget - spent:.2f} Cr")
            m4.metric(label="Foreign Players", value=f"{int(optimal_squad['Is_Foreign'].sum())}/4")
            
            st.markdown("---")

            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                role_budget = display_squad.groupby('Role')['Auction Price (Cr)'].sum().reset_index()
                fig_pie = px.pie(
                    role_budget, 
                    values='Auction Price (Cr)', 
                    names='Role', 
                    title="Budget Allocation by Role", 
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                fig_bar = px.bar(
                    display_squad, 
                    x='Player Name', 
                    y='Fitness_Score' if 'Fitness_Score' in display_squad.columns else 'Fantasy Points', 
                    color='Role',
                    title="Player Fantasy Contributions",
                    text_auto='.2f',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("📋 Your Optimal Squad")
            
            st.dataframe(
               display_squad, 
               use_container_width=True, 
               hide_index=True,
               height=450
            )

            csv = display_squad.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Squad as CSV",
                data=csv,
                file_name='optimal_ipl_squad.csv',
                mime='text/csv',
            )
            
        else:
            st.error("⚠️ Could not find a valid team! This usually happens if your budget is too low for the constraints you set, or if you forced players into the squad that break the team structure. Try adjusting your constraints.")

if __name__ == "__main__":
    main()
