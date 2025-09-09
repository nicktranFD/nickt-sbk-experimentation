import os
import time
import json
import streamlit as st
from scipy.stats import gaussian_kde, chi2_contingency
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, 
    NotebookTask, 
    Source, 
    JobAccessControlRequest, 
    JobPermissionLevel
    )

from sqlalchemy.engine import create_engine

class QueryADP():

    def __init__(self):

        self.connection = self.__set_connection()

    def __set_connection(self):
        access_token = os.getenv("DB_PERSONAL_TOKEN")
        server_hostname = os.getenv("DB_SERVER_HOSTNAME")
        http_path = os.getenv("DB_SERVER_PATH")

        # Create SQLAlchemy engine
        connection = create_engine(f"databricks://token:{access_token}@{server_hostname}?http_path={http_path}")
        return connection
    
    def __tidy_campaign_name(self, campaign_name):
        # Convert 'aadhdad, adadad' to list (split by comma)
        items = [s.strip() for s in campaign_name.split(',')]

        # Create a SQL-friendly string: 'aadhdad','adadad'
        cleaned_campaign_name = ','.join(f"'{item}'" for item in items)
        return cleaned_campaign_name
        
    def get_value_campaign_metric(self, campaign_name):
        """
        
        Parameters
        ----------
            campaign_name : str or list
                Name of a campaign available in 
                discovery_analyst.p_value_customer_list_repository

        Returns
        -------
            df : pandas.DataFrame
        
        Extra information
        -----------------
        Uses following tables:
            discovery_analyst.dashboard_actives
            sb_prd_dp.bv.vw_value_customer_list_repository
        
        >>> campaign_name = 'wc20250120_PUM_ETV1a_DM_Test, wc20250616_HE_Octo_v_HE_Control'

        >>> metric = QueryADP()
        >>> df = metric.get_value_campaign_metric(campaign_name)
        """

        campaign_name = self.__tidy_campaign_name(campaign_name)

        query = f"""
                    SELECT 
                        dh.customer_id
                        , dh.campaign
                        , dh.campaign_active
                        , dh.t_or_c as variant
                        , cs.campaign_start_date
                        , cs.campaign_end_date
                        
                        , SUM(dh.active_week) as actives
                        , SUM(NVL(dh.playerdays,0.000))::float as playerdays 
                        , SUM(nvl(dh.BETS,0.000))::float as BETS   
                        , SUM(nvl(dh.TURNOVER,0.000))::float AS TURNOVER
                        , SUM(nvl(dh.CASH_TURNOVER,0.000))::float AS CASH_TURNOVER
                        , SUM(nvl(dh.CASH_PAYOUT,0.000))::float AS CASH_PAYOUT
                        , SUM(nvl(dh.BB_PAYOUT,0.000))::float AS BB_PAYOUT
                        , SUM(nvl(dh.GROSS_WIN,0.000))::float AS GROSS_WIN
                        , SUM(nvl(dh.COMBINED_GW,0.000))::float AS COMBINED_GW
                        , SUM(nvl(dh.NET_REVENUE,0.000))::float AS NET_REVENUE
                        , SUM(nvl(dh.EML_EXPECTED_NET_REVENUE,0.000))::float EML_EXPECTED_NET_REVENUE
                        , SUM(nvl(dh.EXPECTED_MARGIN_GEN,0.000))::float EXPECTED_MARGIN_GEN
                        , SUM(nvl(dh.FREEBETEGW,0.000))::float as FREEBETEGW
                        , SUM(nvl(dh.EXPECTED_ADJUSTED_GW,0.000))::float EXPECTED_ADJUSTED_GW
                        , SUM(nvl(dh.tot_ETV,0.000))::float tot_ETV
                        , SUM(dh.adj_gw)::float as adj_gw
                        , SUM(dh.GEN_COST)::float as GEN_COST
                        , SUM(dh.expected_gen_cost2)::float as expected_gen_cost2
                        , array_join(collect_set(dh.turnover_outlier), ',') as turnover_outlier
                        , array_join(collect_set(dh.enr_outlier), ',') as enr_outlier
                        , array_join(collect_set(dh.etv_group), ',') as etv_group
                        , array_join(collect_set(dh.lifecycle_stage), ',') as lifecycle_stage
                        , array_join(collect_set(dh.betting_preference), ',') as betting_preference
                        , array_join(collect_set(cs.value_tier), ',') as value_tier
                        , array_join(collect_set(cs.cat_preference), ',') as cat_preference

                    FROM sb_prd_dp.discovery_analyst.p_dashboard_actives as dh
                    LEFT JOIN sb_prd_dp.bv.vw_value_customer_list_repository as cs
                                    ON dh.customer_id=cs.customer_id 
                                    and dh.campaign=cs.campaign_name
                    WHERE dh.campaign in ({campaign_name})
                    GROUP BY 1, 2, 3, 4, 5, 6
                    """
        
        df = pd.read_sql(query, self.connection)

        if df.empty:
            st.error("No data found. Check if Campaign Name exists.")
            
        return df
    
    def get_overlapping_experiment_contingency_count(self, campaign_name):
        """
        Parameters
        ----------
            campaign_name : str or list

            Name of a campaign available in 
            sb_prd_dp.bv.vw_value_customer_list_repository

        Returns
        -------
            df : pandas.DataFrame

        """

        candidate_campaign = self.__tidy_campaign_name(campaign_name)

        query = f"""
        -- Step 1: Get a distinct list of campaigns with their start and end dates
        WITH concurrent_expts AS (
            SELECT DISTINCT 
                campaign_name,
                campaign_start_date,
                campaign_end_date
            FROM sb_prd_dp.bv.vw_value_customer_list_repository
        ),

        -- Step 2: Identify overlapping campaigns for a given candidate campaign
        concurrent_summary_TB_temp AS (
            SELECT 
                a.campaign_name AS Candidate_Campaign_Name,
                a.campaign_start_date AS StartDate1,
                a.campaign_end_date AS EndDate1,
                b.campaign_name AS Reference_Campaign_Name,
                b.campaign_start_date AS StartDate2,
                b.campaign_end_date AS EndDate2,

                -- Calculate the overlapping period between the two campaigns
                GREATEST(a.campaign_start_date, b.campaign_start_date) AS StartDateOverlap,
                LEAST(a.campaign_end_date, b.campaign_end_date) AS EndDateOverlap,

                -- Calculate number of overlapping days
                DATEDIFF(day, GREATEST(a.campaign_start_date, b.campaign_start_date), LEAST(a.campaign_end_date, b.campaign_end_date)) + 1 AS overlap_days,

                -- Total duration of the reference campaign
                DATEDIFF(day, b.campaign_start_date, b.campaign_end_date) + 1 AS total_days

            FROM concurrent_expts a
            JOIN concurrent_expts b 
                ON a.campaign_name <> b.campaign_name
            WHERE (
                -- Check if the campaigns overlap in time
                b.campaign_start_date BETWEEN a.campaign_start_date AND a.campaign_end_date
                OR a.campaign_start_date BETWEEN b.campaign_start_date AND b.campaign_end_date
            )
            AND a.campaign_name in ({candidate_campaign}) -- Filter for the specific candidate campaign
        ),

        -- Step 3: Get customer-level data for each campaign
        p_value_custom AS (
            SELECT 
                campaign_name,
                customer_id,
                t_or_c -- Indicates whether the customer was in the Target or Control group
            FROM sb_prd_dp.bv.vw_value_customer_list_repository
        ),

        -- Step 4: Match customers from the candidate campaign to those in overlapping reference campaigns
        concurrent_summary_cid_TB_temp AS (
            SELECT 
                a.Candidate_Campaign_Name,
                c1.customer_id AS c1_id,
                a.Reference_Campaign_Name,
                c2.customer_id AS c2_id,
                a.StartDateOverlap,
                a.EndDateOverlap,

                -- Determine test group for candidate campaign customer
                CASE 
                    WHEN c1.t_or_c = 'Target' THEN 'target'
                    WHEN c1.t_or_c = 'Control' THEN 'control'
                    ELSE 'not present'
                END AS c1_testGroup,

                -- Determine test group for reference campaign customer (if matched)
                CASE 
                    WHEN c2.t_or_c = 'Target' THEN 'target'
                    WHEN c2.t_or_c = 'Control' THEN 'control'
                    ELSE 'not present'
                END AS c2_testGroup

            FROM concurrent_summary_TB_temp a
            JOIN p_value_custom c1 
                ON a.Candidate_Campaign_Name = c1.campaign_name
            LEFT JOIN p_value_custom c2 
                ON a.Reference_Campaign_Name = c2.campaign_name 
                AND c2.customer_id = c1.customer_id -- Match on customer ID
        )

        -- Step 5: Aggregate results to analyze customer overlap and test/control group distribution
        SELECT 
            a.Candidate_Campaign_Name,
            a.Reference_Campaign_Name,
            a.StartDateOverlap,
            a.EndDateOverlap,
            a.overlap_days,
            a.total_days,

            -- Percentage of overlap duration
            CAST(a.overlap_days * 1.0 / a.total_days AS DOUBLE) AS DaysOverlapPercentage,

            -- Total customers in candidate campaign and how many were also in the reference campaign
            COUNT(c1.customer_id) AS c1Cnt,
            COUNT(c2.customer_id) AS c2Cnt,
            CAST(COUNT(c2.customer_id) AS DOUBLE) / NULLIF(COUNT(c1.customer_id), 0) AS SharedUserPercentage,

            -- Breakdown of shared and non-shared customers by test/control group
            SUM(CASE WHEN c1.t_or_c = 'Control' AND c2.customer_id IS NOT NULL THEN 1 ELSE 0 END) AS Control_In,
            SUM(CASE WHEN c1.t_or_c = 'Target' AND c2.customer_id IS NOT NULL THEN 1 ELSE 0 END) AS Target_In,
            SUM(CASE WHEN c1.t_or_c = 'Control' AND c2.customer_id IS NULL THEN 1 ELSE 0 END) AS Control_Out,
            SUM(CASE WHEN c1.t_or_c = 'Target' AND c2.customer_id IS NULL THEN 1 ELSE 0 END) AS Target_Out,

            -- Cross-group overlap analysis
            SUM(CASE WHEN c1.t_or_c = 'Control' AND c2.t_or_c = 'Control' THEN 1 ELSE 0 END) AS Control_Control,
            SUM(CASE WHEN c1.t_or_c = 'Target' AND c2.t_or_c = 'Control' THEN 1 ELSE 0 END) AS Target_Control,
            SUM(CASE WHEN c1.t_or_c = 'Control' AND c2.t_or_c = 'Target' THEN 1 ELSE 0 END) AS Control_Target,
            SUM(CASE WHEN c1.t_or_c = 'Target' AND c2.t_or_c = 'Target' THEN 1 ELSE 0 END) AS Target_Target

        FROM concurrent_summary_TB_temp a
        JOIN p_value_custom c1 
            ON a.Candidate_Campaign_Name = c1.campaign_name
        LEFT JOIN p_value_custom c2 
            ON a.Reference_Campaign_Name = c2.campaign_name 
            AND c2.customer_id = c1.customer_id
        GROUP BY 
            a.Candidate_Campaign_Name,
            a.Reference_Campaign_Name,
            a.StartDateOverlap,
            a.EndDateOverlap,
            a.overlap_days,
            a.total_days
        """

        df = pd.read_sql(query, self.connection)

        if df.empty:
            st.error("No data found. Check if Campaign Name exists.")
            
        return df
        

def db_create_job(
    task_key=""
    ,notebook_params=None
    ,variants_data=None
    ,description=""):

    w = WorkspaceClient()

    # We need to cache Control/Target data because
    # they are too big to submit as a json to the job
    variants_cache_path = os.getenv("variants_cache_path")
    w.files.upload(variants_cache_path, json.dumps(variants_data), overwrite=True)

    try:
        w.clusters.ensure_cluster_is_running(os.getenv("cluster_id"))
    except:
        print("Your connection to databricks isn't configured correctly.")

    run_response = w.jobs.run_now(
        job_id = int(os.getenv("job_id")),
        notebook_params = notebook_params
        )
    run_id = run_response.run_id
    
    # Wait for the job to complete
    while True:
        run_status = w.jobs.get_run(run_id=run_id)
        run_message = str(run_status.state.life_cycle_state).split('.')[-1]
        if run_message in ['TERMINATED','SKIPPED','INTERNAL_ERROR']:
            st.write("Analysis completed.")
            break        
        st.write("Waiting for analysis to complete...")
        time.sleep(30)


    # Retrieve the output of each individual task run
    task_runs = w.jobs.get_run(run_id=run_id).tasks
    for task_run in task_runs:
        task_run_id = task_run.run_id
        run_output = w.jobs.get_run_output(run_id=task_run_id)
    
    result = run_output.notebook_output.result if run_output.notebook_output else "No Output"
    
    result = json.loads(result)
    return result

def put_logo():
    with st.sidebar:
        c1, c2 = st.columns(2)
        c1.image("abatarapp/gallery/references/abatar_logo.png", use_container_width=True)
        c2.image("abatarapp/gallery/references/powered_by_DS.png", use_container_width=True)


class ChiSqr_Contingency:
    def __init__(self):
        """
        Initializes the ChiSqr class.
        """
        pass

    def __rowwise_chi2(self, row, var_names, alpha=0.05, header='Traffic Interaction'):
        """
        Performs a chi-squared test on the given row of data.

        Adds small values to the table to avoid division by zero errors.

        Args:
            row (pandas.Series): A row of data to perform the chi-squared test on.
            var_names (list): A list of variable names to use in the chi-squared test.
            alpha (float): The significance level for the chi-squared test. Defaults to 0.05.
            header (str): The header for the chi-squared test results. Defaults to 'Traffic Interaction'.
        """

        table = [
                [row[var_names[0]] + 1e-10, row[var_names[1]] + 1e-10],
                [row[var_names[2]] + 1e-10, row[var_names[3]] + 1e-10]
                ]
        chi2, p, dof, expected = chi2_contingency(table)
        
        sig = 'Dependence' if p < alpha else 'Fail to Reject Dependence'
    
        return pd.Series(
            {header + ' ' +'Chi-Squared': float(chi2)
             , header + ' ' + 'p_value': p
             , header:sig})
        
    def get_traffic_interaction(self, sub_df):
        """
        Performs a chi-squared test on the given sub_df of data to compute traffic interaction

        Args:
            sub_df (pandas.DataFrame): A sub_df of data to perform the chi-squared test on.

        Returns:
            pandas.DataFrame: A DataFrame containing the results of the chi-squared test.
        """
        var_names = ['Control_In', 'Target_In', 'Control_Out', 'Target_Out']

        t_interaction = lambda row: self.__rowwise_chi2(row, var_names, header='Traffic Interaction')
        results = sub_df.apply(t_interaction, axis=1).round(3)
        return results

    def get_cross_traffic_interaction(self, sub_df):
        """
        Performs a chi-squared test on the given sub_df of data to compute cross traffic interaction.

        Args:
            sub_df (pandas.DataFrame): A sub_df of data to perform the chi-squared test on.
        
        Returns:
            pandas.DataFrame: A DataFrame containing the results of the chi-squared test.
        """
        var_names = ['Control_Control', 'Target_Control', 'Control_Target', 'Target_Target']

        crosst_interaction = lambda row: self.__rowwise_chi2(row, var_names, header='Cross-Traffic Interaction')
        results = sub_df.apply(crosst_interaction, axis=1).round(3)
        return results


def plot_uplift(x, xleft, xcen, xright):
    """_summary_

    Args:
        x (array): full distribution of metric e.g. uplift
        xleft (float): left bound of confidence interval
        xcen (float): central estimate (mean/median) of metric e.g. uplift
        xright (float): right bound of confidence interval

    Returns:
        fig, (ax0, ax1) (_type_): figure and axes for the plot
    """
    plt.style.use(['tableau-colorblind10'])
    fig = plt.figure(figsize=(9,4))

    gs = GridSpec(2, 1, height_ratios=[9, 1])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # removing frame
    ax0.spines['bottom'].set_visible(False)

    for ax in [ax0, ax1]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])

    ax0.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-ticks
    ax1.xaxis.set_major_formatter(PercentFormatter())
    ax1.tick_params(axis='both', labelsize=8)

    kde = gaussian_kde(x, bw_method=0.4)
    xr = np.linspace(x.min(), x.max(), 100)
    kde_yr = kde(xr)

    ax0.plot(xr, kde_yr, lw=2, color='C0', label='Uplift Distribution')
    ybins, xbins,_ = ax0.hist(x, density=True
                            , bins=100
                            , histtype='step'
                            , color='C0', alpha=0.25)
    ax0.vlines(0, ymax=kde(0).max(), ymin=0
            , color='r', lw=2, ls='dashed', label='Reference Uplift = 0%', alpha=0.8)
    ax0.vlines(xcen, ymax=kde(xcen).max(), ymin=0
            , color='#ff7f00', lw=2, label='Observed Uplift')
    ax0.vlines(xleft, ymax=kde(xleft).max(), ymin=0
            , color='#ff7f00', lw=2, ls='dashed', label='Uplift CI')
    ax0.vlines(xright, ymax=kde(xright).max(), ymin=0
            , color='#ff7f00', lw=2, ls='dashed')
    ax0.text(xcen, -0.01, f'{xcen:2.0f}%', color='#ff7f00', ha='center', fontsize=8)
    ax0.text(xleft, -0.01, f'{xleft:2.0f}%', color='#ff7f00', ha='center', fontsize=8)
    ax0.text(xright, -0.01, f'{xright:2.0f}%', color='#ff7f00', ha='center', fontsize=8)

    ax0.legend(ncol=4, loc='upper left', fontsize=8, bbox_to_anchor=(0.,-0.4), frameon=False)

    # Uplift Bar/Interval
    ax1.errorbar(xcen, 0.05
                , xerr=[[xcen-xleft],[xright-xcen]]
                , color='#ff7f00', lw=2, capsize=8, capthick=3)
    ax1.plot(xcen, 0.05, 's', ms=5, color='#ff7f00')
    ax1.plot(0, 0.05, 's', ms=5, color='r', alpha=0.8)
    ax0.set_title("Incremental Uplift [%]", fontsize=10, fontweight='bold', color='#1C75BC')
    ax1.minorticks_on()
    return fig, (ax0, ax1)

def plot_variants_dist(
    control: np.ndarray,
    target: np.ndarray,
    control_label: str = 'Control',
    target_label: str = 'Target',
    bins: int = 50,
    remove_zero: bool = False,
    xlabel:str = None,
    ax:plt.axes = None,
) -> plt.axes:
    """Plot variants 1D distribution
          a. regular 1D hist
          b. regular 1D hist for non zero elements only
          c. log transformed 1D hist 

    Parameters
    ----------
    control : np.ndarray
        
    target : np.ndarray
        
    remove_zero : bool, optional
        default False
    bins : int, optional
        by default 50
    xlabel : str, optional
        by default None
    ax : plt.axes, optional
        by default None

    Returns
    -------
    plt.axes
    """
    
    if ax == None:
        _, ax = plt.subplots(num=1)

    if remove_zero == True:
        _control = control[control > 0]
        _target = target[target > 0]

        print(f"Control non-null data: {100*len(_control)/len(control):0.2f}%")
        print(f"Target non-null data: {100*len(_target)/len(target):0.2f}%")

        control = _control
        target = _target

    ax.hist(
        control,
        label=control_label,
        histtype="step",
        bins=bins,
        lw=2,
    )
    ax.hist(
        target,
        label=target_label,
        histtype="step",
        bins=bins,
        lw=2,
    )
    ax.legend()

    if xlabel:
        ax.set_xlabel(xlabel)
        
    return ax


def check_column_is_numeric(df, column_name):
    """
    Function to check if a column is numeric
    """
    
    # Check if the column is numeric
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Metric {column_name} is not of numeric type!")
