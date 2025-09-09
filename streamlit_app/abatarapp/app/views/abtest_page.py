import streamlit as st
import matplotlib.pyplot as plt
from abatar import PermutationTest, Bootstrap, abtest, plot_uplift

import altair as alt
import pandas as pd
import numpy as np
from . import utils
import os

def create_page():
    """
    A/B Test Page"""
    st.empty()

    if "Target" not in st.session_state:
        st.error("Target must be set. Check Data page.")
    
    try:
        target = st.session_state['Target'].tolist()
    except AttributeError:
        st.error(f"Target must be set. Check Data page.")


    if "Control" not in st.session_state:
        st.error("Control must be set. Check Data page.")

    try:
        control= st.session_state['Control'].tolist()
    except AttributeError:
        st.error(f"Control must be set. Check Data page.")

    nsample = len(control + target)
    
    if "metric" not in st.session_state:
        st.error("Metric name must be selected. Check Data page.")
        
    metric = st.session_state["metric"]

    st.header(f":blue[A/B Test Analysis]")
    st.text_area("Use this page to conduct an A/B test to assess whether there is a statistically significant difference between the Target and Control groups under the null hypothesis of no effect.",
                "The underlying methods used to test the hypothesis are resampling-based:\n" "      a. Permutation Test for hypothesis testing, and \n"
                "      b. Bootstrapping for estimating uplift and confidence intervals.\n" 
                "Refer to the Definitions section below for descriptions of the parameters used in this analysis.", height=120
        )


    with st.expander("**Definitions**"):
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
        
    st.markdown(f'<h5 style="color: #FAA71C;">{metric.upper()}</h5><hr>', unsafe_allow_html=True)
 
    # Permutation Tests Parameters
    c1, c2, c3 = st.columns(3)

    with c1:
        alpha = st.slider(r"Significance level ($\alpha$)"
                          ,min_value=0, max_value=10
                          ,value=5
                          ,step=1
                          ,format="%d%%"
                          )
        alpha = alpha/100

    with c2:
        alternative = st.selectbox(
            "Alternative Hypothesis"
            , ['two-sided', 'greater', 'less']
            , index=0
            )

    with c3:
        test_stat = st.selectbox(
            "Test Statistics"
            , ['mean', 'median']
            , index=0
            )
        
    # Bootstrap Parameters
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        n_permu = st.number_input(
            "Number of Permutation Runs"
            , min_value=5000, max_value=40000
            , value=20000 if nsample < 25000 else 5000
            , step=1000, format="%d")

    with c2:
        straps = st.number_input(
            "Number of Bootstraps"
            , min_value=5000, max_value=500000
            , value=25000 if nsample < 30000 else 5000
            , step=2500, format="%d")

    with c3:
        bootstrap_test_stat = st.selectbox(
            "Bootstrap Test Stat"
            , ['median', 'mean']
            , index=1)

    with c4:
        random_seed = st.number_input(
            "Random Seed"
            , min_value=1, max_value=500000
            , value=42
            , step=1, format="%d")
                

    ## Classical Tests
    if st.button(":red[Run Hypothesis Test]"):

        # DEMO suite - runs on App Cluster
        if st.session_state.demo==True:
            st.write("Running in DEMO mode.")
            np.random.seed(random_seed)

            # Permutaton Test
            pt = PermutationTest(
                    control
                    , target
                    , alpha=alpha
                    , n_permutations= n_permu
                    , test_stat = test_stat
                    , alternative = alternative
                    )
            obs_stat, pvalue, null_dist, reject_H0 = pt.run_test(verbose=True)

            res_permu_table = pt._generate_result_message(pvalue, obs_stat)[:-250]
            res_statement = f"Because, if the null hypothesis is true, the chance of observing a difference between Target and Control as extreme as {obs_stat:.3f} is {pvalue*100:2.2f}%, "

            # Bootstrap Uplift and Uncertainties
            bt = Bootstrap(
                control, 
                target, 
                test_stat = bootstrap_test_stat,
                straps = straps
                )
            bt_result = bt.run_test(verbose=True)
            uplift_percent_dist, uplift_percent_est = bt_result[0]
            uplift_diff_dist, uplift_diff_est = bt_result[1]

            res_boot_table = bt._generate_result_message(uplift_percent_est, uplift_diff_est)

        # Submit job to DB abatar cluster
        else:
            st.write("Pushing execution to Databricks Cluster.")

            notebook_params = {'n_permu': n_permu
                                , 'alpha': alpha
                                , 'alternative': alternative
                                , 'test_stat': test_stat
                                , 'seed': random_seed
                                , 'bootstrap_stat': bootstrap_test_stat
                                , 'straps':straps
                                , 'variants_cache_path': os.getenv('variants_cache_path')
                                }
            
            variants_dict = {'control':control, 'target':target}
            
            result = utils.db_create_job(
                task_key = "permutation_and_bootstrap"
                ,notebook_params = notebook_params
                ,variants_data = variants_dict
                ,description = "Run Classical Permutation Test and Bootstrapping for Uplift Calculation."
                )
            
            obs_stat = result['obs_stat']
            pvalue = result['pvalue']
            null_dist = result['null_dist']
            reject_H0 = result['reject_H0']
            res_permu_table = result['permutation_report']
            
            res_statement = f"Because, if the null hypothesis is true, the chance of observing a difference between Target and Control as extreme as {obs_stat:.3f} is {pvalue*100:2.2f}%,"

            uplift_percent_dist, uplift_percent_est = result['uplift_percent_dist'], result['uplift_percent_est']
            uplift_diff_dist, uplift_diff_est = result['uplift_diff_dist'], result['uplift_diff_est']
            res_boot_table = result['bootstrap_report']
                
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

        c1, c2, c3 = st.columns([0.1,0.8,0.1])
        with c2:
            c2a, c2b = st.columns([0.2,0.8])
            with c2a:
                st.metric(label="UPLIFT"
                        , value=f"{uplift_diff_est[1]:2.2f}"
                        , delta=f"{uplift_percent_est[1]:2.0f}%")
            with c2b:
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
        c1, c2, c3 = st.columns([0.1,0.8,0.1])
        with c2:
            _, c2b, c_ = st.columns([0.1,0.8,0.1])
            with c2b:
                fig, ax = utils.plot_uplift(np.array(uplift_percent_dist)
                                , uplift_percent_est[0]
                                , uplift_percent_est[1]
                                , uplift_percent_est[2])
                st.pyplot(fig)

        c1, c2, c3 = st.columns([0.15,0.7,0.15])
        with c2:
            c2a, c2b = st.columns([0.55,0.45])
            with c2a:
                st.subheader(":blue[Significance Report]")
                st.write(res_permu_table)
                
            with c2b:
                st.subheader(":blue[Uplift Measurements]")
                st.write(res_boot_table)
