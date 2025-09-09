import streamlit as st
import pandas as pd

from . import utils

def create_page():
    """
    Test Interaction Page"""
    
    st.empty()

    st.subheader(":blue[Tests Interaction]", divider="gray")

    st.text_area("Use this page to quantify the extent to which tests with overlapping customers impact each other.",
                "The underlying methods used are:\n"
                "      a. Chi-sqr contingency: solely based on customer counts , and \n"
                "      b. More robust estimates using business metrics of overlapping customers [Work-in-progress].\n" 
                "Currently this page source data only from Sportsbet discovery_analyst.p_dashboard_actives ADP table.", height=120
        )

    if 'candidate_campaign' not in st.session_state:
        st.session_state.candidate_campaign = "wc20250120_PUM_ETV1a_DM_Test"

    def update_text_input():
        st.session_state.candidate_campaign = st.session_state.text_input

    candidate_campaign = st.text_input(
        "Candidate Campaign Name(s)"
        , value=st.session_state.candidate_campaign
        , on_change=update_text_input
        , placeholder="e.g. wc20250120_PUM_ETV1a_DM_Test, wc20250616_HE_Octo_v_HE_Control"
        , key="text_input")
    
    if candidate_campaign != "":
        
        # Fetch Interaction Data between a candidate and reference tests
        query_inst = utils.QueryADP()
        df = query_inst.get_overlapping_experiment_contingency_count(candidate_campaign)
        
        # Compute ChiSqr from the ordered pair of Control/Target Ins/Outs Targets/Controls
        chisqr = utils.ChiSqr_Contingency()
        
        result1 = chisqr.get_traffic_interaction(df[['Control_In', 'Target_In', 'Control_Out', 'Target_Out']])

        result2 = chisqr.get_cross_traffic_interaction(df[['Control_Control', 'Target_Control', 'Control_Target', 'Target_Target']])

        result_df = pd.concat([df, result1, result2], axis=1)

        result_df.sort_values(['Candidate_Campaign_Name', 'Traffic Interaction p_value'], inplace=True)

        # Showing important columns first
        preferred = ['Candidate_Campaign_Name', 'Reference_Campaign_Name'
                     , 'Traffic Interaction Chi-Squared', 'Traffic Interaction p_value'
                     , 'Traffic Interaction', 'Cross-Traffic Interaction Chi-Squared'
                     , 'Cross-Traffic Interaction p_value', 'Cross-Traffic Interaction'
                     ]

        cols = result_df.columns
            
        st.write("Traffic Interaction Between Candidate and Reference Tests")
        st.dataframe(result_df[cols[cols.isin(preferred)].tolist() + cols[~cols.isin(preferred)].tolist()])