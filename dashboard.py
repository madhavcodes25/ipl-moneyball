import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_cleaning import prepare_data
from src.optimizer import calculate_fantasy, optimize_team

st.set_page_config(page_title="IPL Moneyball AI", page_icon="🏏", layout="wide")

@st.cache_data
def load_data():
    return prepare_data()

def main():
    raw_data = load_data()
    player_list = sorted(raw_data['Player Name'].dropna().unique().tolist())

    with st.sidebar:
        st.header("Strategy Room 🏏")
        st.markdown("Fine-tune your auction constraints.")
        
        with st.form("strategy_form"):
            budget = st.slider("Total Budget (Crores)", 20, 100, 50)
            role_focus = st.selectbox("Team Strategy", ["Balanced", "Batting Heavy", "Bowling Heavy"])
            
            st.markdown("---")
            st.subheader("Retentions")
            must_include_players = st.multiselect(
                "Force include up to 11 players:",
                options=player_list,
                max_selections=11,
                help="The AI will build the rest of the team around these retained players."
            )
            
            st.markdown("---")
            generate_btn = st.form_submit_button("Draft Perfect Team", type="primary", use_container_width=True)

    st.title("🏆 AI-Powered IPL Team Selector")
    st.caption(f"Building the best mathematically possible team using the '{role_focus}' strategy.")
    st.divider()

    if not generate_btn:
        st.info("👈 Adjust your parameters in the Strategy Room and click 'Draft Perfect Team' to begin.")
        return

    with st.spinner("Crunching the numbers and simulating auction bids..."):
        
        scored_data = calculate_fantasy(raw_data, strategy=role_focus)
        
        optimal_squad, total_score = optimize_team(
            scored_data, 
            budget=budget,
            must_include=must_include_players
        )
        
        if optimal_squad is not None and not optimal_squad.empty:
            spent = optimal_squad['Cost_Cr'].sum()
            
            st.subheader("📊 Team Analytics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(label="Total Fantasy Score", value=f"{total_score:.2f}")
            m2.metric(label="Funds Spent", value=f"₹{spent:.2f} Cr")
            m3.metric(label="Funds Remaining", value=f"₹{budget - spent:.2f} Cr")
            m4.metric(label="Foreign Players", value=f"{int(optimal_squad['Is_Foreign'].sum())}/4")
            
            st.markdown("---")

            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                role_budget = optimal_squad.groupby('Role')['Cost_Cr'].sum().reset_index()
                fig_pie = px.pie(
                    role_budget, 
                    values='Cost_Cr', 
                    names='Role', 
                    title="Budget Allocation by Role", 
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                fig_bar = px.bar(
                    optimal_squad, 
                    x='Player Name', 
                    y='Fitness_Score' if 'Fitness_Score' in optimal_squad.columns else 'Fantasy_Score', 
                    color='Role',
                    title="Player Fantasy Contributions",
                    text_auto='.2f',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("📋 Your Optimal Squad")
            
            st.dataframe(
               optimal_squad, 
               use_container_width=True, 
               hide_index=True,
               height=450
            )

            csv = optimal_squad.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Squad as CSV",
                data=csv,
                file_name='optimal_ipl_squad.csv',
                mime='text/csv',
            )
            
        else:
            st.error("⚠️ Could not find a valid team! This usually happens if your budget is too low for the players you forced into the squad. Try increasing your budget.")

if __name__ == "__main__":
    main()
