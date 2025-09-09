import streamlit as st 
from abatar import PermutationTest, SimulatePower

def create_page():
    """
    Read Me Page
    """
    st.empty()
    st.title("FAQ")

    st.subheader("Permutation Test")
    st.markdown("""The main approach used in ABatar for A/B testing is [Permutation Test](https://en.wikipedia.org/wiki/Permutation_test).
                 It is an exact statistical hypothesis test making use of the proof by contradiction. 
                A permutation test involves two or more samples. The null hypothesis is that all samples come from the same distribution. 
                Under the null hypothesis, the distribution of the test statistic is obtained by calculating 
                all possible values of the test statistic under possible rearrangements of the observed data.
                """
                )
    st.write(PermutationTest)
    st.subheader("Power Simulation")
    st.write(SimulatePower)

    # st.markdown(
    # """
    # Q1.  
    # Q2.  
    # Q3.  
    # """
    # )

    st.title("Contact Us")
    st.markdown("""
                ABatar package ([Github Link](https://github.com/Sportsbet-Internal/dp-services/tree/feature/MLMDG-196-ds-abatar-in-ECS/apps/ds-abatar)) is developed and maintained by Marcus Data Science Team. Report bug or send general query to the slack channel #abatar-abtest-analytics or directly to prajwal.kafle@sportsbet.com.au.
                """
                )
