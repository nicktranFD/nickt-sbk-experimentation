import streamlit as st
import streamlit_option_menu as stmenu
import pandas as pd
import numpy as np
from views import (home_page
                   , data_page
                   , abtest_page
                   , power_sim_page
                   , tests_interaction_page
                   , readme_page
                   , bayesian_page
                   , utils)
import matplotlib.pyplot as plt 

st.set_page_config(layout="wide"
                   , page_title="ABatar"
                   , page_icon="abatarapp/gallery/references/abatar_logo.png")


with st.sidebar:   
    selected = stmenu.option_menu(
        "",
        options = ["Home"
                   , "Data"
                   , "A/B Test"
                   , "Power Simulation"
                   , "Tests Interaction"
                   , "Bayesian Test"
                   , "Read Me"],
        icons = ["house"
                 , "cloud-arrow-up"
                 , "shuffle"
                 ,  "bar-chart"
                 , "intersect"
                 , "boombox"
                 , "book"
                 ],
        menu_icon = "arrows-angle-contract",
        default_index = 0,
        styles={"icon": {"color": "#FAA71C", "font-size": "25px","font-weight":"bold"},
                "nav-link":{"font-weight":"bold", "font-size": "20px"},
                "nav-link-selected": {"background-color": "#0067AB"}
                }
)

if selected == "Home":
    home_page.create_page()

if selected == "Data":
    utils.put_logo()
    data_page.create_page()

if selected == "A/B Test":
    utils.put_logo()
    abtest_page.create_page()

if selected == "Power Simulation":
    utils.put_logo()
    power_sim_page.create_page()

if selected == "Tests Interaction":
    utils.put_logo()
    tests_interaction_page.create_page()

if selected == "Bayesian Test":
    utils.put_logo()
    bayesian_page.create_page()

if selected == "Read Me":
    utils.put_logo()
    readme_page.create_page()