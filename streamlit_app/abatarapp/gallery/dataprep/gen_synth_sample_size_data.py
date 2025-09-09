# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Metric Extraction

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC with tiers as (
# MAGIC   SELECT
# MAGIC         distinct
# MAGIC         da.customer_id,
# MAGIC         (CASE
# MAGIC         WHEN ap.premium_tier_id IS NOT NULL AND cg.customer_group = 'Premium Team Managed' THEN 'PM'
# MAGIC         WHEN ap.premium_tier_id IS NOT NULL AND cg.customer_group_team = 'Managed' THEN 'PM' 
# MAGIC         WHEN ap.premium_tier_id IS NOT NULL THEN 'PUM'
# MAGIC         ELSE
# MAGIC         (CASE
# MAGIC             WHEN etv.tier_cd = 1 THEN 'ETV1'
# MAGIC             WHEN etv.tier_cd = 2 THEN 'ETV2'
# MAGIC             WHEN etv.tier_cd = 3 THEN 'ETV3'
# MAGIC             WHEN etv.tier_cd = 4 THEN 'ETV4'
# MAGIC         END)
# MAGIC         END) cat_value_tier
# MAGIC     FROM bv.vw_etv_daily_account_tier etv
# MAGIC     INNER JOIN bv.vw_dim_account da
# MAGIC         ON etv.account_id = da.account_id
# MAGIC     LEFT JOIN bv.vw_account_premium_tier ap
# MAGIC         ON etv.account_id = ap.account_id
# MAGIC         AND etv.report_dt >= ap.report_start_darwin_date
# MAGIC         AND etv.report_dt <= ap.report_end_darwin_date
# MAGIC     LEFT JOIN bv.vw_dim_customer_group_premium_tier_history cg
# MAGIC         ON da.customer_id = cg.customer_id
# MAGIC         AND etv.report_dt >= cg.ctl_effective_start_darwin_dts
# MAGIC         AND etv.report_dt <= cg.ctl_effective_end_darwin_dts
# MAGIC     where 
# MAGIC         etv.report_dt = date_add(current_date, -1)
# MAGIC         and is_trade_account <> 'Y'
# MAGIC         and is_test_account <> 'Y'
# MAGIC )
# MAGIC
# MAGIC select 
# MAGIC distinct
# MAGIC   da.customer_id
# MAGIC   , tiers.cat_value_tier
# MAGIC   , mean(stake_aud)::float as cash_turnover
# MAGIC from bv.vw_dim_account da
# MAGIC left join bv.vw_account_metric_daily vm on vm.account_id=da.account_id
# MAGIC left join tiers on vm.customer_id = tiers.customer_id
# MAGIC where
# MAGIC   da.is_trade_account <> 'Y'
# MAGIC   and da.is_test_account <> 'Y'
# MAGIC   and p_ctl_transaction_darwin_dt between date_add(current_date, -45) and date_add(current_date, -1)
# MAGIC group by 1, 2

# COMMAND ----------

df = _sqldf.toPandas()

# COMMAND ----------

df.dropna(inplace=True)

# COMMAND ----------

df.info(show_counts=True)

# COMMAND ----------

df.groupby(['cat_value_tier']).describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Outlier trimming

# COMMAND ----------

# Cash TurnOver
def metric_by_tier(atier):
    data = df.loc[df['cat_value_tier'].isin(atier), 'cash_turnover']

    # winsorizing
    lower_bound = np.percentile(data, 0)  # 1st percentile
    upper_bound = np.percentile(data, 99)  # 99th percentile
    trimmed_data = np.clip(data, lower_bound, upper_bound)

    mean, std = trimmed_data.mean(), trimmed_data.std()
    return mean, std

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define Sample Size Calculator

# COMMAND ----------

def sample_size_two_sample_ztest(std, base_mean, effect_size):
    """
    std: variance of the distribution
    base_mean: mean of the control sample
    effect_size: stipulated % uplift on base_mean expected from the experiment
    """
    delta = effect_size*base_mean/100
    sample_size_per_group = int(15.68*std**2/delta**2)
    total_sample_size = 2*sample_size_per_group #for target+control
    return total_sample_size

# COMMAND ----------

# MAGIC %md
# MAGIC #### Sample Size Estimation

# COMMAND ----------

result = {
    ('Cash T/O', 'PUM'):{},
    ('Cash T/O', 'Mass'):{},
    ('Cash T/O', 'Overall'):{}
    }
effect_size = [2.5, 5, 7.5, 10] #expected percentage uplift on Control mean
for isample in ('PUM', 'Mass', 'Overall'):
    for ieffect_size in effect_size:
        if isample=='PUM':
            atier = ('PUM',)
        elif isample=='Mass':
            atier = ('ETV1', 'ETV2', 'ETV3', 'ETV4')
        else:
            atier = ('ETV1', 'ETV2', 'ETV3', 'ETV4', 'PUM')

        mean, std = metric_by_tier(atier)
        sample_est = sample_size_two_sample_ztest(std, mean, ieffect_size)
        result[('Cash T/O',isample)][f'{ieffect_size}%'] = sample_est

print('Sample Size Estimation for the Cash Turn Over')
print('For different effect sizes')
pd.DataFrame.from_dict(result).T

# COMMAND ----------

# Save data

ndata = df.shape[0]

df['gaussian_metric'] = np.random.normal(100, 10, ndata)
df['conversion_metric'] = np.random.binomial(1, 0.15, ndata)

# COMMAND ----------

df_sample = df.sample(frac=0.02)
df_sample.shape

# COMMAND ----------

df_sample.to_csv("sample_size_est_synth_data.csv", index=False)

# COMMAND ----------

df_sample

# COMMAND ----------

display(df)

# COMMAND ----------


