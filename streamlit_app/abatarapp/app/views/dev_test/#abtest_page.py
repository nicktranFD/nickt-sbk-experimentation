import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import numpy as np
from . import utils

def create_page():
    st.empty()

    db_details = utils.DatabricksCreds()

    cluster_id = st.session_state['cluster_id']

    if "Target" not in st.session_state:
        st.error("Target must be set. Check data page.")

    target = st.session_state['Target']
    target_aslist = str(target.tolist())

    if "Control" not in st.session_state:
        st.error("Control must be set. Check data page.")

    control= st.session_state['Control']
    control_aslist = str(control.tolist())

    if "metric" not in st.session_state:
        st.error("Metric name must be selected. Check data page.")
        
    metric = st.session_state["metric"]

    st.header(f":blue[A/B Test Analysis]")
    st.markdown(f'<h4 style="color: #FAA71C;">{metric.upper()}</h4><hr>', unsafe_allow_html=True)

    with st.expander("**Parameters**"):
        st.markdown("""
                **Number of Permutations**: The total number of possible re-arrangements or orderings of the sample data used in permutation tests to assess the distribution of the test statistic under the null hypothesis.\n\n

                **Significance level**: The threshold probability (usually 0.05) used to determine whether the test result is statistically significant.\n\n  

                **Alternative Hypothesis**: The hypothesis that there is an effect or difference, contrasting the null hypothesis. It is what the test aims to provide evidence for.\n\n  
             
                **Test Statistics**: A value calculated from sample data (e.g. mean) used to assess the null hypothesis. It is compared against a distribution under the null hypothesis to determine significance.\n\n
             
                **Number of Bootstraps**: The number of times the data is resampled with replacement to generate new samples to estimate uplift.\n\n 
             
                **Bootstrap Test Stat**: The value of the test statistic calculated for each resampled bootstrap sample. It is used to create an empirical distribution of the uplift.\n\n
             
                **Random Seed**: A value used to initialize the random number generator. It ensures that the results are reproducible by producing the same random values across different runs for both Permutation and Bootstrapping methods.
             """
             )
        
    
    
    # Permutation Tests Parameters
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        n_permu = st.number_input(
            "Number of Permutations"
            , min_value=2500, max_value=15000
            , value=2500
            , step=1000, format="%d")

    with c2:
        alpha = st.number_input(
            r"Significance level $\alpha$"
            , min_value=0.001, max_value=0.1
            , value=0.05
            , step=0.005, format="%3f")

    with c3:
        alternative = st.selectbox(
            "Alternative Hypothesis"
            , ['two-sided', 'greater', 'less']
            , index=0
            )

    with c4:
        test_stat = st.selectbox(
            "Test Statistics"
            , ['mean', 'median']
            , index=0
            )
        
    # Bootstrap Parameters
    c1, c2, c3 = st.columns(3)

    with c1:
        straps = st.number_input(
            "Number of Bootstraps"
            , min_value=5000, max_value=500000
            , value=5000
            , step=1000, format="%d")

    with c2:
        bootstrap_test_stat = st.selectbox(
            "Bootstrap Test Stat"
            , ['median', 'mean']
            , index=1)

    with c3:
        random_seed = st.number_input(
            "Random Seed"
            , min_value=1, max_value=500000
            , value=42
            , step=1, format="%d")
                
    ## Permutation Test
    if st.button(":red[Run Hypothesis Test]"):
        st.write("Pushing execution to Databricks Cluster.")

        result = utils.db_create_job(
            f"abatar-app-permutation-test-{metric}"
            ,notebook_path = db_details.permu_nb_path
            ,task_key="permutation"
            ,notebook_params= {'n_permu': n_permu
                            , 'alpha': alpha
                            , 'alternative': alternative
                            , 'target': target_aslist
                            , 'control': control_aslist
                            , 'test_stat': test_stat
                            , 'seed': random_seed
                            }
            ,notebook_source="WORKSPACE"
            ,description=""
            ,existing_cluster_id=cluster_id
            )
        obs_stat = result['obs_stat']
        pvalue = result['pvalue']
        null_dist = result['null_dist']
        reject_H0 = result['reject_H0']
        res_permu_table = result['report']
        
        res_statement = f"Because, if the null hypothesis is true, the chance of observing a difference between Target and Control as extreme as {obs_stat:.3f} is {pvalue*100:2.2f}%, "


        # Uplift Reporting

        result = utils.db_create_job(
        f"abatar-app-bootstrapping-{metric}"
        ,notebook_path = db_details.bootstrap_nb_path
        ,task_key="bootstrap"
        ,notebook_params={'straps': straps
                            , 'test_stat': bootstrap_test_stat
                            , 'target': target_aslist
                            , 'control': control_aslist
                            , 'random_seed': random_seed
                            }
        ,notebook_source="WORKSPACE"
        ,description=""
        ,existing_cluster_id=cluster_id
        )

        uplift_percent_dist, uplift_percent_est = result['uplift_percent_dist'], result['uplift_percent_est']
        uplift_diff_dist, uplift_diff_est = result['uplift_diff_dist'], result['uplift_diff_est']
        res_boot_table = result['report']
            
        st.subheader(":blue[Test Results]")
        st.markdown("<hr>", unsafe_allow_html=True)

        # Uplift Chart
        if reject_H0 == 0:
            metric_color = '#E34A33'
            nhst_msg = "Failed to Reject the Null Hypothesis"
            res_statement += ', which is too high to rule out as random variation.' 
        else:
            metric_color = '#4C9F70'
            nhst_msg = "Reject the Null Hypothesis"
            res_statement += ', which is very unlikely.'

        c1, c2 = st.columns([1,6])
        with c1:
            st.metric(label="UPLIFT"
                      , value=f"{uplift_diff_est[1]:2.2f}"
                      , delta=f"{uplift_percent_est[1]:2.0f}%")
        with c2:
            with st.expander('Summary', expanded=True):
                st.markdown(f'<span style="color: {metric_color}; font-weight: bold; font-size: 20px;">{nhst_msg}:</span> {res_statement}', unsafe_allow_html=True)

        # Show uplift
        st.markdown(f"""
            <style>
            [data-testid="block-container"] {{
                padding-left: 2rem;
                padding-right: 2rem;
                padding-top: 1rem;
                padding-bottom: 0rem;
                margin-bottom: -7rem;
            }}
            [data-testid="stVerticalBlock"] {{
                padding-left: 0rem;
                padding-right: 0rem;
            }}
            [data-testid="stMetric"] {{
                background-color: #FAFAFA;
                text-align: center;
                padding: 15px 0;
                border: 2px solid {metric_color}; /* Add a green border */
                border-radius: 8px; /* Optional: Rounded corners */
            }}
            [data-testid="stMetricLabel"] {{
            display: flex;
            justify-content: center;
            align-items: center;
            }}
            [data-testid="stMetricDeltaIcon-Up"] {{
                position: relative;
                left: 38%;
                -webkit-transform: translateX(-50%);
                -ms-transform: translateX(-50%);
                transform: translateX(-50%);
            }}
            [data-testid="stMetricDeltaIcon-Down"] {{
                position: relative;
                left: 38%;
                -webkit-transform: translateX(-50%);
                -ms-transform: translateX(-50%);
                transform: translateX(-50%);
            }}
            </style>
            """, unsafe_allow_html=True)
                    
        # Results Visualisation
        c1, c2 = st.columns([3,1])
        with c1:
            fig, ax = utils.plot_uplift(np.array(uplift_percent_dist)
                            , uplift_percent_est[0]
                            , uplift_percent_est[1]
                            , uplift_percent_est[2])
            st.pyplot(fig)

        with c2:
            st.subheader(":blue[Significance Report]")
            st.write(res_permu_table)

            st.subheader(":blue[Uplift Report]")
            st.write(res_boot_table)