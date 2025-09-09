# Databricks notebook source
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
import numpy as np
import scipy.stats as st


# COMMAND ----------

x = np.random.normal(loc=1, scale=4, size=10000)
xleft, xcen, xright = np.percentile(x,[2.5,50,97.5])


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

kde = st.gaussian_kde(x, bw_method=0.4)
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
ax0.text(xcen, -0.01, f'{xcen:2.0f}%', color='#ff7f00', ha='center')
ax0.text(xleft, -0.01, f'{xleft:2.0f}%', color='#ff7f00', ha='center')
ax0.text(xright, -0.01, f'{xright:2.0f}%', color='#ff7f00', ha='center')

ax0.legend(ncol=4, loc='upper left', fontsize=10, bbox_to_anchor=(0.,-0.4), frameon=False)

# Uplift Bar/Interval
ax1.errorbar(xcen, 0.05
            , xerr=[[xcen-xleft],[xright-xcen]]
            , color='#ff7f00', lw=2, capsize=8, capthick=3)
ax1.plot(xcen, 0.05, 's', ms=5, color='#ff7f00')
ax1.plot(0, 0.05, 's', ms=5, color='r', alpha=0.8)
ax0.set_title("Incremental Uplift")
ax1.minorticks_on()

# COMMAND ----------

%pip install databricks-sdk --upgrade

dbutils.library.restartPython()

# COMMAND ----------

from databricks_sdk import WorkspaceClient

w = WorkspaceClient()
existing_cluster_id = "1126-032422-ppt23ob"

permu_nb_path = "/Users/prajwal.kafle@sportsbet.com.au/ds-abatar-app/abatarapp/app/views/task_permutation_spark_test"

cluster_id = w.clusters.ensure_cluster_is_running(existing_cluster_id)

run = w.jobs.submit(run = "test_abatar",
                    tasks = [
                        jobs.SubmitTask(
                            existing_cluster_id = cluster_id,
                            notebook_task = jobs.NotebookTask(permu_nb_path),
                            task_key = "test_abatar",)
                    ])

run.results()

# COMMAND ----------


