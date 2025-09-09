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

target = rd.normal(105, 10, 150000)
control = rd.normal(100, 10, 150000)

target.shape

# COMMAND ----------

n_permu = 5000
alpha = 0.05
alternative = 'less'
test_stat = 'mean'
seed = 42

# COMMAND ----------

rd.seed(seed)
pt = PermutationTest(
                control
                , target
                , alpha=alpha
                , n_permutations= n_permu
                , test_stat = test_stat
                , alternative = alternative
                , batch=1000
                )
obs_stat, pvalue, null_dist, reject_H0 = pt.run_test(verbose=True)

report = pt._generate_result_message(pvalue, obs_stat)[:-250]

# COMMAND ----------

obs_stat, pvalue, reject_H0

# COMMAND ----------

print(report)

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

def permutation_test_spark(control
                           , target
                           , statistic
                           , n_resamples
                           , alternative
                           ):
    """
    Perform a permutation test with options for alternative hypotheses.
    
    Parameters:
        control (pd.Series): The control group values.
        target (pd.Series): The target group values.
        num_permutations (int): The number of permutations to perform.
        alpha (float): The significance level.
        alternative (str): The type of alternative hypothesis ('two-sided', 'greater', 'less').
    
    Returns:
        observed_stat (float): The observed test statistic.
        p_value (float): The p-value for the test.
        null_distribution (list): The null hypothesis distribution.
        reject_null (bool): Whether to reject the null hypothesis based on alpha.
    """
    # Calculate the original observed test statistic (e.g., difference in means)
    observed_stat = statistic(target) - statistic(control)
    
    # Convert the pandas series to Spark RDD for parallelism
    rdd = spark.sparkContext.parallelize(range(n_resamples))
    
    # Function to perform one permutation
    def permutation_stat_rdd(_):
        # Shuffle the values in the target column using np.random.permutation
        shuffled_values = np.random.permutation(np.concatenate([control, target]))
        
        # Reassign the shuffled values to the "Control" and "Target" groups
        shuffled_control = shuffled_values[:len(control)]
        shuffled_target = shuffled_values[len(control):]
        
        # Calculate the new test statistic after shuffling
        perm_stat = statistic(shuffled_target) - statistic(shuffled_control)
        return perm_stat
    
    # Perform permutations in parallel
    null_dist = rdd.map(permutation_stat_rdd).collect()
    
    # Calculate p-value based on alternative hypothesis
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(null_dist) >= np.abs(observed_stat))
    elif alternative == 'greater':
        p_value = np.mean(null_dist >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(null_dist <= observed_stat)
    else:
        raise ValueError("Alternative hypothesis must be 'two-sided', 'greater', or 'less'")
       
    # Return results
    return observed_stat, p_value, null_dist

# Null Hypothesis Significance Test
class PermutationTestSpark:
    """2 sample, 1 or 2 sided hypothesis testing using non-parametric permutation method:
    Ho: no difference between mean(Target) and mean(Control)
    H1: there is a difference between mean(Target) and mean(Control)

    Notes:
    Suppose data contains two samples, e.g. control and target
    The data are pooled (concatenated), then randomly split and assigned to either
    the first or second sample, and the test statistic is calculated.
    This process is performed repeatedly (n_iters times)
    generating a distribution of the statistic under the null hypothesis.
    The observed statistic of the original data is compared to
    this distribution to determine the p-value.

    Ref: https://docs.scipy.org/doc//scipy/reference/generated/scipy.stats.permutation_test.html

    Methods
    -------

    run_test(verbose=True)

    Examples
    --------
    Pseudo-data

    >>> target = np.array([8.53, 8.52, 8.01, 7.99, 7.93, 7.89, 7.85, 7.82, 7.8])
    >>> control = np.array([7.85, 7.73, 7.58, 7.4, 7.35, 7.3, 7.27, 7.27, 7.23])

    Instantiate PermutationTest
    
    >>> pt = PermutationTest(control, target, n_permutations=5000, test_stat='mean', axis=-1)
    
    Run test
    
    >>> obs_stat, pvalue, null_dist, reject_H0 = pt.run_test(verbose=True)
    """

    def __init__(
        self,
        control: np.ndarray,
        target: np.ndarray,
        n_permutations: int = 5000,
        batch: int = None,
        test_stat: str = 'mean',
        alpha: float = 0.05,
        alternative: str = "two-sided",
        vectorized: bool = False,
        axis: int = 0,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        control : np.ndarray
            Control sample data.
        target : np.ndarray
            Target sample data.
        n_permutations : int, optional
            Number of permutations, by default 5000.
        batch : int, optional
            Batch size for parallel computation, by default None.
        test_stat : str, optional
            Type of test statistic to use, by default 'mean'.
        alpha : float, optional
            Significance level, by default 0.05.
        alternative : str, optional
            Alternative hypothesis, by default "two-sided".
        vectorized : bool, optional
            Whether to use vectorized computation, by default False.
        axis : int, optional
            Axis for computation, by default 0.
        random_state : int, optional
            Random seed for reproducibility, by default 42.
        """
        
        self.control = control
        self.target = target
        self.n_permutations = n_permutations
        self.batch = batch
        self.alpha = alpha
        self.random_state = random_state
        self.vectorized = vectorized
        self.alternative = alternative
        self.axis = axis

        if test_stat == 'mean':
            self.func = np.mean
        elif test_stat == 'median':
            self.func = np.median
        else:
            raise AttributeError("test stat undefined.")
        self.test_stat = test_stat

    def run_test(self, verbose: bool = False) -> tuple:
        """
        Run permutation test.

        Parameters
        ----------
        verbose : bool, optional
            Whether to display verbose output, by default False.

        Returns
        -------
        tuple
            Observed statistic, p-value, null distribution, and reject_H0.
        """
        
        res = permutation_test_spark(
            self.control,
            self.target,
            statistic=self.func,
            n_resamples=self.n_permutations,
            alternative=self.alternative,
        )

        obs_stat, pvalue, null_dist = res

        reject_H0 = int(pvalue < self.alpha)

        if verbose:
            message = self._generate_result_message(pvalue, obs_stat)
            # logging.info(message)

        return obs_stat, pvalue, null_dist, reject_H0

    def _generate_result_message(self, pvalue: float, obs_stat: float) -> str:
        """
        Generate a result message based on the p-value and observed statistic.

        Parameters
        ----------
        pvalue : float
            The p-value of the test.
        obs_stat : float
            The observed test statistic.

        Returns
        -------
        str
            A message describing the test result.
        """
        alternative_desc = (
        "greater than" if self.alternative == "greater" else
        "less than" if self.alternative == "less" else
        "not equal to")

        test_conclusive = (
            "Conclusive" if pvalue < self.alpha else
            "Inconclusive")
        
        difference_significance = (
            "Stat. significant" if pvalue < self.alpha else
            "Not stat. significant")

        if self.alternative == "greater":
            H0 = "<"
            # HA = ">="
        elif self.alternative == "less":
            H0 = ">"
            # HA = "<="
        else:
            H0 = "="
            # HA = "!="

        if pvalue >= self.alpha:
            null_hypothesis_rejection = "Fail to reject"
        else:                
            null_hypothesis_rejection = "Reject"            

        null_hypothesis_line1 = (
            f"Null Hypothesis: mean(Target) {H0} mean(Control)")

        message = (
            f"\n{'-' * 81}\n"
            f"{'Permutation Test Attributes':<50}| {'Outputs':<30}\n"
            f"{'-'*50}|{'-'*30}\n"
            f"{'Null Hypothesis Test':<50}| {test_conclusive:<30}\n"
            f"{'Difference between the Target and Control':<50}| {difference_significance:<30}\n"
            f"{null_hypothesis_line1:<50}| {null_hypothesis_rejection:<30}\n"
            f"{'p-value':<50}| {pvalue:<30.6f}\n"
            f"{'Observed stat, i.e., mean(Target) - mean(Control)':<50}| {obs_stat:<30.3f}\n"
            f"\nIf we repeat the experiment many times, the likelihood of observing the difference\n"
            f"between Target and Control as extreme as {obs_stat:.3f} is {100*pvalue:.2f}%."
            f"\n{'-' * 82}\n"
        )
        return message
    


# COMMAND ----------

rd.seed(seed)

pt = PermutationTestSpark(
                control
                , target
                , alpha=alpha
                , n_permutations= n_permu
                , test_stat = test_stat
                , alternative = alternative
                )
obs_stat, pvalue, null_dist, reject_H0 = pt.run_test(verbose=True)

report2 = pt._generate_result_message(pvalue, obs_stat)[:-250]
print(report2)

# COMMAND ----------

print(report)

# COMMAND ----------

alternative

# COMMAND ----------


