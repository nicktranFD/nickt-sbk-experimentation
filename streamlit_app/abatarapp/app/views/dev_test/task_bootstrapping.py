# Databricks notebook source
# %pip install --upgrade databricks-sdk
%pip install --upgrade numba

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from abatar import Bootstrap
import json
import numpy.random as rd

# COMMAND ----------

dbutils.widgets.text("control", "[3.0,4,5]")
dbutils.widgets.text("target", "[5.0,6,7]")
dbutils.widgets.text("straps", "1000")
dbutils.widgets.text("test_stat", "median")
dbutils.widgets.text("seed", "42")

control = eval(dbutils.widgets.get("control"))
target = eval(dbutils.widgets.get("target"))
straps = int(dbutils.widgets.get("straps"))
test_stat = dbutils.widgets.get("test_stat")
seed = int(dbutils.widgets.get("seed"))

# COMMAND ----------

rd.seed(seed)

bt = Bootstrap(
    control,
    target,
    test_stat = test_stat,
    straps = straps
    )
bt_result = bt.run_test()
uplift_percent_dist, uplift_percent_est = bt_result[0]
uplift_diff_dist, uplift_diff_est = bt_result[1]

report = bt._generate_result_message(uplift_percent_est, uplift_diff_est)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    'uplift_percent_dist':uplift_percent_dist.tolist()
    , 'uplift_percent_est':uplift_percent_est
    , 'uplift_diff_dist':uplift_diff_dist.tolist()
    , 'uplift_diff_est':uplift_diff_est
    , 'report':report}))
