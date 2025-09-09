import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import utils
import os

def create_page():
    """
    Data Page
    """
    st.empty()

    st.subheader(":blue[Load Campaign Data]", divider="gray")
    
    st.text_area("Use this page to provide the data required to conduct an A/B test", 
                "At a minimum, the dataset must include a variant column and one or more metric columns (referred to as metric_name).\n"
                "You can load the data using one of the following methods:\n"
                "       a. Upload a .csv file containing the campaign data, making sure metrics are numeric data-type or\n"
                "       b. Specify campaign name(s) available in the discovery_analyst.p_dashboard_actives ADP table."
        , height=120
        )
    @st.cache_data
    def load_data(filename):
        if filename:
            return pd.read_csv(filename)
        return None
    
    @st.cache_data
    def query_campaign(campaign_name):
        query_inst = utils.QueryADP()
        df = query_inst.get_value_campaign_metric(campaign_name)
        return df
    
    if "data" not in st.session_state:
        default_path = "abatarapp/gallery/dataprep/sample_test_data.csv"
        data = pd.read_csv(default_path)
        st.session_state.data = data

    # Upload Data OR Campaign Query Tabs
    c1, c2 = st.columns([0.5,0.5], border=True, gap="medium")
    
    with c1:
        uploaded_file = st.file_uploader("File (Minimum columns required are: variant, and metric(s).)", type="csv", label_visibility='visible')
    
    if uploaded_file is not None:
        st.session_state.data = load_data(uploaded_file)

    # Campaign Names and related Configurations
    # Function to update session state values
    def update_text_input():
        st.session_state.campaign_name = st.session_state.text_input

    # Initialize session state if not already done
    if 'campaign_name' not in st.session_state:
        st.session_state.campaign_name = ""

    with c2:
        campaign_name = st.text_input(
            "Campaign Name"
            , value=st.session_state.campaign_name
            , on_change=update_text_input
            , placeholder="e.g. wc20250120_PUM_ETV1a_DM_Test, wc20250616_HE_Octo_v_HE_Control"
            , key="text_input")
        
        if campaign_name!='':
            data = query_campaign(campaign_name)

            c2a, c2b, c2c = st.columns(3)

            with c2a:
                campaign_active = st.selectbox(
                    "Campaign Active Flag"
                    , ["1. Pre-Test", "2. In-Test", "3. Post-Test"]
                    , index=1
                    , key="campaign_selectbox"
                    )
                
            data = data.query(f"campaign_active=='{campaign_active}'")
            
            with c2b:
                value_tier = st.selectbox(
                    "Value Tier"
                    , ['All'] + list(data['value_tier'].unique())
                    , index=0
                    , key="tier_selectbox"
                    )

            if value_tier!='All':
                data = data.query(f"value_tier=='{value_tier}'")

            with c2c:
                outlier = st.selectbox(
                    "Outlier"
                    , ['None', 'turnover_outlier', 'enr_outlier']
                    , index=0
                    , key="outlier_selectbox"
                    )

            if outlier!='None':
                data = data.query(f"{outlier}=='Not outlier'")

            st.session_state.data = data

    df = st.session_state.data
    df.fillna(0)
    df = df.apply(pd.to_numeric, errors='ignore')

    # Load id, variant and metrics attributes
    # variant_names = np.sort(df['variant'].unique())
    variant_names = df['variant'].unique()
    metrics = df.loc[:, ~df.columns.isin(['variant'
                                          , 'id'
                                          , 'customerid'
                                          , 'customer_id'])]
    metrics_name = metrics.select_dtypes(include=np.number).keys()

    c1, c2, c3 = st.columns([1,6,1])
    with c2:
        st.dataframe(df.head(50), use_container_width=True)
        st.download_button(label="Download Data as CSV",
            data=df.to_csv(index=False),
            file_name='data.csv',
            mime='text/csv'
        )

    if df.empty:
        st.error("Data not loaded. Check Campaign Name or Uploaded File.")
        st.stop()
         
    grouped_stats = df.groupby(["variant"])[metrics_name].describe().round()
    st.dataframe(grouped_stats, use_container_width=True)


    # Select variants for analysis
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(":blue[Control]", divider="grey")
        sel_control = st.selectbox(
            'Control',
            variant_names,
            index=0,
            label_visibility='hidden',
            placeholder="Select Control attribute...",
        )

        control = df.query(f"variant=='{sel_control}'")

    with c2:
        st.subheader(":blue[Target]", divider="grey")
        sel_target = st.selectbox(
            'Target',
            variant_names,
            index=1,
            label_visibility='hidden',
            placeholder="Select Target attribute...",
        )

        target = df.query(f"variant=='{sel_target}'")
        
    # Select Metric for analysis
    with c3:
        st.subheader(":blue[Metric]", divider="grey")
        sel_metric = st.selectbox(
            'Metric',
            metrics_name,
            index=0,
            label_visibility='hidden',
            placeholder="Select a metric to analyse...",
        )

    # Check if metric is numeric
    try:
        utils.check_column_is_numeric(control, sel_metric)
    except ValueError as ve:
        st.error(f"ValueError: {ve}")
    except TypeError as te:
        st.error(f"TypeError: {te}")

    try:
        utils.check_column_is_numeric(target, sel_metric)
    except ValueError as ve:
        st.error(f"ValueError: {ve}")
    except TypeError as te:
        st.error(f"TypeError: {te}")

    # Cache variables to persist in other pages
    if "Target" not in st.session_state:
        st.session_state["Target"] = ""
    else:
        st.session_state["Target"] = target[sel_metric]

    if "Control" not in st.session_state:
        st.session_state["Control"] = ""
    else:
        st.session_state["Control"] = control[sel_metric]

    if "metric" not in st.session_state:
        st.session_state["metric"] = ""
    else:
        st.session_state["metric"] = sel_metric

    st.subheader(":blue[Data Preview]", divider="gray")
    c1, c2, c3 = st.columns([1, 4, 3])
    with c2:
        st.subheader("")
        plt.style.use(['tableau-colorblind10'])
        fig, ax = plt.subplots()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])

        utils.plot_variants_dist(
            st.session_state['Control'],
            st.session_state['Target'], 
            control_label = sel_control,
            target_label = sel_target,
            ax=ax, 
            xlabel=st.session_state['metric'].upper()
            )
        ax.set_ylabel('')
        st.pyplot(fig)

    with c3:
        st.subheader("")
        
        # T/C proportion
        sample_size = df.groupby(['variant']).size() 
        sample_sum = df.shape[0]
        t_c_proportion = pd.concat([sample_size, 
                                    (100*sample_size/sample_sum).round(1)], 
                                    axis=1, keys=['Sample Size', 'Percentage'])
        st.write(t_c_proportion)

