import streamlit as st
from abatar import SimulatePower
import altair as alt
import pandas as pd
import numpy as np
from scipy.stats import binom, norm


def create_page():
    """
    Power Simulation Page
    """
    st.empty()
    st.header(":blue[Sample Size Calculator]", divider="gray")

    st.text_area("Use this page to estimate the required sample size for an A/B test to reliably detect a statistically meaningful difference between variants.", 
        "Two methods are available:\n"
        "    a. For a simple calculation, provide the baseline summary statistics (conversion rate, mean, stdev etc) for the metric of interest.\n"
        "    b. For a more robust estimation, upload the historical distribution of the metric, aggregated at the same level at which the test will be conducted."
        , height=90
        )

    with st.expander("**Definitions**"):
        st.markdown(
            """
                **Target Fraction**: The proportion of the total population out of *Sample Size* assigned to the target group (e.g., 25% in treatment). \n\n
                **Effect Size**: The magnitude of the difference or relationship being tested (e.g., a 5% improvement in the metric). Larger effect sizes are easier to detect. \n\n
                **Significance Level**: The probability threshold (e.g., α = 0.05) below which the null hypothesis is rejected, representing the Type I error rate. \n\n
                **No. of Simulations**: The number of times the power simulation is repeated to estimate the power of the test. A higher number improves reliability. \n\n
                **No. of Permutations**: The number of rearrangements of the data performed in permutation tests to calculate the null distribution. \n\n
                **Random Seed**: A value used to initialise the random number generator. It ensures that the results are reproducible by producing the same random values across different runs for both Permutation and Bootstrapping methods. \n\n
                **Power**: The probability that the test correctly detects a true effect when one exists. A common target for power is 80% or higher.\n\n
                **Alternative**: The alternative hypothesis, which specifies the direction of the effect. It can be one of 'two-sided', 'greater', or 'less'.\n\n
                **Metric**: The key measurement used to evaluate the effect (e.g., conversion rate, stake, ETV). \n\n
                **Segment**: The column used to stratify the data. \n\n
                **Tier/Class**: Limit estimations to tier/class level.
             """
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        relative_uplift = st.slider(
            r"Incremental Uplift / Effect Size (%)",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            format="%d%%",
        )
        relative_uplift = relative_uplift / 100

    with c3:
        alpha = st.slider(
            r"Significance level ($\alpha$)",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            format="%d%%",
        )
        alpha = alpha / 100

    with c2:
        target_frac = st.slider(
            r"Target Fraction",
            min_value=0.05,
            max_value=0.95,
            value=0.5,
            step=0.1,
            format="%2f",
        )

    with c4:
        alternative = st.selectbox(
            "Alternative Hypothesis"
            , ['two-sided', 'greater', 'less']
            , index=0
            )

    st.session_state['metric_val'] = None

    metac1, metac2 = st.columns(2)

    with metac1:
        st.subheader(":blue[Simple Calculator]", divider="gray")
        st.write("Method based on the key parameters of the distribution of the metric. Need to specify binary (conversions) or non-binary (turnovers, revenues etc) nature of the metric.")

        rng = np.random.default_rng(1234)

        metric_type = st.selectbox(
            "Metric Type",
            ['Conversion/Binary', 'Non-binary'],
            index=0,
            placeholder="Specify whether metric is conversion or not.",
        )

        if metric_type=='Conversion/Binary':
            baseline_p = st.number_input(
                    r"Baseline Conversion Rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=None,
                )


            if baseline_p is not None:
                st.session_state['metric_val'] = binom.rvs(n=1, p=baseline_p
                                                           , random_state=rng
                                                           , size=100000
                                                           )
            else:
                st.session_state['metric_val'] = None

        elif metric_type=='Non-binary':
            baseline_mean = st.number_input(
                    r"Baseline Mean of the Metric",
                    value=None,
                )

            baseline_std = st.number_input(
                    r"Baseline stdev of the Metric",
                    value=None,
                )

            if baseline_mean is not None and baseline_std is not None:
                st.session_state['metric_val'] = norm.rvs(baseline_mean, baseline_std, size=100000
                            , random_state=rng)
            else:
                st.session_state['metric_val'] = None

    with metac2:        
        st.subheader(":blue[Distribution Based Calculator]", divider="gray")
        st.write("Method based on the full distribution of the metric.")

        # -------- Upload Data File
        uploaded_file = st.file_uploader(
            "File (Provide column of historical metric(s) aggregated at levels actual test data would be.)",
            type="csv",
            label_visibility="visible",
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.session_state['metric_val'] = None
        
            st.write(df.head())

            # -------- Permutation Tests Parameters
            c1, c2, c3 = st.columns(3)

            metrics = df.loc[
                :, ~df.columns.isin(["variant", "id", "customerid", "customer_id"])
            ]

            with c1:
                metric = st.selectbox(
                    "Metric",
                    metrics.select_dtypes(include=np.number).keys(),
                    index=0,
                    placeholder="Select a metric to analyse...",
                )

            with c2:
                seg_key = df.keys().tolist()
                seg_key.insert(0, "NA")

                segment = st.selectbox(
                    "Segment",
                    seg_key,
                    index=0,
                    placeholder="Select column to analyse...",
                )

            with c3:
                tier = "NA"
                if segment in df.keys():
                    seg_vals = sorted(df[segment].unique())

                    tier = st.selectbox(
                        "Tier/Class",
                        seg_vals,
                        index=0,
                        placeholder="Select class/tier to analyse...",
                    )

            np.random.seed(42)

            if tier == "NA":
                st.session_state['metric_val'] = df[metric].values
            else:
                st.session_state['metric_val'] = df.loc[df[segment] == tier, metric].values


            # c1, c2, c3 = st.columns(3)
            # with c1:
            #     niters = st.number_input(
            #         r"No. of Simulations"
            #         , min_value=500, max_value=10000
            #         , value=1000
            #         , step=500, format="%d")

            # with c2:
            #     npermu = st.number_input(
            #         r"No. of Permutations"
            #         , min_value=500, max_value=10000
            #         , value=500
            #         , step=500, format="%d")

            # with c3:
            #     random_seed = st.number_input(
            #         r"Random Seed"
            #         , min_value=1, max_value=100000
            #         , value=42
            #         , step=1, format="%d")

    niters, npermu = 1000, 500

    st.subheader(":blue[Sample Estimate]", divider="gray")

    if st.session_state['metric_val'] is not None:
        res = SimulatePower(
            st.session_state['metric_val'],
            sample_size=10,
            effect_size=relative_uplift,
            target_frac=target_frac,
            alpha=alpha,
            n_iters=niters,
            n_permutations=npermu,
            alternative=alternative
        )

        res_print = res.get_parametric_est()
        test_sample_size = int(res_print.split()[-1])
        test_name = res_print.split()[0].upper()
        st.markdown(
            f"##### To observe a statistically significant results [{test_name} Power = 80%]:") 
        st.markdown(f"##### Overall Sample Size Required: :green[{test_sample_size}]")
        target_size = int(target_frac*test_sample_size)
        st.markdown(f"##### With Target Size: :blue[{target_size}] and Control Size: :blue[{test_sample_size - target_size}]")
    """
    # --------- Run power simulation estimation
    st.subheader(f":grey[Power Simulation]")
    c1, c2, c3 = st.columns(3)
    with c1:
        simu_sample_size = st.number_input(
            "Sample",
            min_value=10,
            max_value=1500000,
            value=test_sample_size,
            step=1000,
            format="%d",
            label_visibility="hidden",
        )

    if st.button(":red[Run Simulation]"):
        st.write(
            f"Overall sample size required to observe statistically significant result:"
        )
        res.sample_size = simu_sample_size
        # res.simulate()
        st.write(res)
    """