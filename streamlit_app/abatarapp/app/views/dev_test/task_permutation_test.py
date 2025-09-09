# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC %pip install --upgrade numba

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from abatar import PermutationTest
import json
import numpy.random as rd

# COMMAND ----------

dbutils.widgets.text("n_permu", "1000")
dbutils.widgets.text("alpha", "0.05")
dbutils.widgets.text("alternative", "two-sided")
dbutils.widgets.text("test_stat", "mean")
dbutils.widgets.text("control", "[3.0,4,5]")
dbutils.widgets.text("target", "[5.0,6,7]")
dbutils.widgets.text("seed", "42")

n_permu = int(dbutils.widgets.get("n_permu"))
alpha = float(dbutils.widgets.get("alpha"))
alternative = dbutils.widgets.get("alternative")
test_stat = dbutils.widgets.get("test_stat")
control = eval(dbutils.widgets.get("control"))
target = eval(dbutils.widgets.get("target"))
seed = int(dbutils.widgets.get("seed"))

# COMMAND ----------

rd.seed(seed)
pt = PermutationTest(
                control
                , target
                , alpha=alpha
                , n_permutations= n_permu
                , test_stat = test_stat
                , alternative = alternative
                , batch = 1000
                )
obs_stat, pvalue, null_dist, reject_H0 = pt.run_test(verbose=True)

report = pt._generate_result_message(pvalue, obs_stat)[:-250]

# COMMAND ----------

dbutils.notebook.exit(json.dumps({'obs_stat':obs_stat.tolist()
                                  , 'pvalue':pvalue
                                  , 'null_dist':null_dist.tolist()
                                  , 'reject_H0':reject_H0
                                  , 'report':report}))
