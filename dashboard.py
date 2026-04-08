import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_cleaning import prepare_data
from src.optimizer import calculate_fantasy, optimize_team
# Configure the main Streamlit page layout and metadata
st.set_page_config(page_title="IPL Moneyball", page_icon="🏏", layout="wide")

@st.cache_data
def load_data():
    """Caches and loads the cleaned IPL data to prevent redundant processing on app re-runs."""
    return prepare_data()

def main():
    # Load data and generate an alphabetically sorted list of unique player names
    raw_data = load_data()
    player_list = sorted(raw_data['Player Name'].dropna().unique().tolist())
    # Set up the interactive sidebar menu for user inputs
    with st.sidebar:
        st.header("Strategy Room 🏏")
        st.markdown("Fine-tune your auction constraints.")
        
        # Allow user to choose between flexible or strict team composition rules
        constraint_mode = st.radio(
                "Select Constraint Mode", 
                ["Flexible (Auto-balance)", "Strict (Custom Roles)"],
                help="Flexible ensures min 5 bowlers and 6 batters. Strict lets you pick exact counts."
            )
        
        # Slider to define the maximum auction budget
        budget = st.slider("Total Budget (Crores)", 20, 120, 100)
        # Dropdown to select the overarching team fantasy scoring strategy
        role_focus = st.selectbox("Team Strategy", ["Balanced", "Batting Heavy", "Bowling Heavy"])
            
        # Visual divider before the next section    
        st.markdown("---")
        st.subheader("Team Composition")
            
        # Default fallback values for Flexible mode    
        num_batters, num_bowlers, num_all_rounders = 5, 4, 2
        num_pacers, num_spinners = 3, 2
        # If Strict mode is selected, show exact number inputs for team roles
        if constraint_mode == "Strict (Custom Roles)":
                st.caption("Must exactly equal 11 players")
                # Use columns to place number inputs side-by-side to save vertical space
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
        # If Flexible mode is selected, just show an informational message            
        else:
                st.info(
                    "🤖 **Auto-balance Active**\n\n"
                    "The algorithm will automatically build the most optimal squad ensuring at least:\n"
                    "- 6 Batting options\n"
                    "- 5 Bowling options\n"
                    "- 1 Wicket Keeper"
                )
        # Immediate validation: The core roles must sum up to exactly 11 players
        if (num_batters + num_bowlers + num_all_rounders) != 11:
                    st.error("🚨 Constraint Error: The sum of Batters, Bowlers, All-Rounders, and Wicket Keepers must exactly equal 11.")
                    return
            
            
        st.markdown("---")
        st.subheader("Retentions")
        # Multiselect box allowing the user to pick up to 11 players to force into the team
        must_include_players = st.multiselect(
                "Force include up to 11 players:",
                options=player_list,
                max_selections=11,
                help="Will build the rest of the team around these retained players."
            )

        retention_prices = {}
        # If the user selects retained players, show inputs to customize their retention price
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
        # Multiselect box to completely ban specific players from the draft pool
        must_exclude_players = st.multiselect(
                "Ignore these players:",
                options=player_list,
                help="These players will be completely ignored by the optimizer."
            )
        st.markdown("---")
        # st.expander creates a collapsible section to hide advanced settings and save space
        with st.expander("🛠️ Custom Expected Prices"):
            st.caption("Adjust expected prices for players in the auction pool.")
            # Filter the player list so users can't override prices for players they already retained or ignored
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
                    # Add a unique key to the input to prevent Streamlit widget conflicts
                    custom_auction_prices[player] = st.number_input(
                        f"{player} Expected Price", 
                        min_value=0.0, 
                        max_value=float(budget), 
                        value=default_cost, 
                        step=0.25,
                        key=f"auction_price_{player}" 
                    )  
        st.markdown("---")

        # The main action button to trigger the backend algorithm
        generate_btn = st.button("Draft Perfect Team", type="primary", use_container_width=True)
    # --- MAIN PAGE AREA ---
    st.title("🏆 Data-Driven IPL Team Selector")
    st.caption(f"Building the best mathematically possible team using the '{role_focus}' strategy.")
    st.divider()
    # If the user hasn't clicked the generate button yet, show an instruction and stop execution
    if not generate_btn:
        st.info("👈 Adjust your parameters in the Strategy Room and click 'Draft Perfect Team' to begin.")
        return


    # --- PRE-OPTIMIZATION VALIDATION CHECKS ---
    # 1. Check if the user tried to both retain AND exclude the exact same player
    conflict_players = set(must_include_players).intersection(set(must_exclude_players))
    if conflict_players:
        st.error(f"🚨 Logic Error: You cannot both retain and ignore the same player(s): {', '.join(conflict_players)}")
        return
    # 2. Final verification of the 11-player limit
    if (num_batters + num_bowlers + num_all_rounders) != 11:
        st.error("🚨 Constraint Error: The sum of Batters, Bowlers, All-Rounders, and Wicket Keepers must exactly equal 11.")
        return
    # 3. Check if bowling constraints exceed total requested bowlers    
    if (num_pacers + num_spinners) > (num_bowlers + num_all_rounders):
        st.warning("⚠️ Logic Warning: You have requested more Pacers and Spinners than total available bowling options (Bowlers + All-Rounders). The optimizer may fail.")
    # 4. Validate retained player rules (Overseas limit & Budget limit)
    if must_include_players:
        retained_df = raw_data[raw_data['Player Name'].isin(must_include_players)]
        # Handle potential variations in column names dynamically
        cost_col = 'Cost_Cr' if 'Cost_Cr' in retained_df.columns else 'Price'
        foreign_col = 'Is_Foreign' if 'Is_Foreign' in retained_df.columns else 'Overseas'
        # Calculate total spent on retentions
        budget_spent_retained = sum(retention_prices.values())
        overseas_count_retained = retained_df[foreign_col].sum() if foreign_col in retained_df.columns else 0
        # IPL Rule: Max 4 overseas players allowed
        if overseas_count_retained > 4:
            st.error(f"🚨 Rule Violation: You have retained {int(overseas_count_retained)} overseas players. The maximum allowed is 4.")
            return 
        # Check if retentions bankrupt the team    
        if budget_spent_retained > budget:
            st.error(f"🚨 Budget Exceeded: Your retained players cost ₹{budget_spent_retained:.2f} Cr, which is over your ₹{budget} Cr limit.")
            return 
            

    # --- RUN THE OPTIMIZATION ENGINE ---
    # st.spinner shows a loading animation while the backend code runs
    with st.spinner("Crunching the numbers and simulating auction bids..."):
        # Step 1: Assign fantasy points to all players based on the selected strategy
        scored_data = calculate_fantasy(raw_data, strategy=role_focus)
        # Step 2: Run the linear programming solver to find the mathematically perfect 11
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
        # --- DISPLAY RESULTS ---
        if optimal_squad is not None and not optimal_squad.empty:
            # Apply the user's custom prices to the final dataframe for accurate display
            for player, custom_price in retention_prices.items():
                optimal_squad.loc[optimal_squad['Player Name'] == player, 'Cost_Cr'] = custom_price

            for player, custom_price in custom_auction_prices.items(): 
                optimal_squad.loc[optimal_squad['Player Name'] == player, 'Cost_Cr'] = custom_price
            # Calculate total money spent
            spent = optimal_squad['Cost_Cr'].sum()
            # Clean up column names for the user-facing table
            display_squad = optimal_squad.rename(columns={
                'Cost_Cr': 'Auction Price (Cr)',
                'Fantasy_Score': 'Fantasy Points',
                'Is_Foreign': 'Overseas Player'
            })
            # Top-level metric cards
            st.subheader("📊 Team Analytics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(label="Total Fantasy Points", value=f"{total_score:.2f}") 
            m2.metric(label="Funds Spent", value=f"₹{spent:.2f} Cr")
            m3.metric(label="Funds Remaining", value=f"₹{budget - spent:.2f} Cr")
            m4.metric(label="Foreign Players", value=f"{int(optimal_squad['Is_Foreign'].sum())}/4")
            
            st.markdown("---")

            # --- DATA VISUALIZATIONS ---
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Aggregate spending by role for the pie chart
                role_budget = display_squad.groupby('Role')['Auction Price (Cr)'].sum().reset_index()
                fig_pie = px.pie(
                    role_budget, 
                    values='Auction Price (Cr)', 
                    names='Role', 
                    title="Budget Allocation by Role", 
                    hole=0.4, # Creates a donut chart style
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                # Bar chart comparing individual player fantasy scores
                fig_bar = px.bar(
                    display_squad, 
                    x='Player Name', 
                    y='Fitness_Score' if 'Fitness_Score' in display_squad.columns else 'Fantasy Points', 
                    color='Role',
                    title="Player Fantasy Contributions",
                    text_auto='.2f', # Show data labels formatted to 2 decimals
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
            # --- FINAL DATA TABLE & EXPORT ---
            st.subheader("📋 Your Optimal Squad")
            # Render an interactive dataframe
            st.dataframe(
               display_squad, 
               use_container_width=True, 
               hide_index=True,
               height=450
            )
            # Convert dataframe to CSV for download
            csv = display_squad.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Squad as CSV",
                data=csv,
                file_name='optimal_ipl_squad.csv',
                mime='text/csv',
            )
        # Fallback if the optimizer failed to find a valid combination    
        else:
            st.error("⚠️ Could not find a valid team! This usually happens if your budget is too low for the constraints you set, or if you forced players into the squad that break the team structure. Try adjusting your constraints.")
# Python standard entry point
if __name__ == "__main__":
    main()
