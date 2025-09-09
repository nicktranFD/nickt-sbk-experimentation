import time
import streamlit as st
from databricks.sdk import WorkspaceClient
import os

def create_page():
    """
    Home Page
    """
    st.empty()

    # Set Header
    # Custom CSS to center the text
    st.markdown("""
        <style>
        .custom-subheader {
            text-align: center;  /* Center align the text */
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="custom-subheader" style="color: #0066CC; font-size: 35px;">EXPERIMENT DESIGN AND ANALYSIS TOOL</h3>', unsafe_allow_html=True)
    st.write("")
    st.write("")

    # First paragraph
    st.markdown(
    '<p class="custom-subheader" style="font-size: 25px; color: gray;">'
    'Welcome to our in-house A/B Testing web interface.<br>' 
    'To get started, explore training materials and example codes provided <a href="https://github.com/prajwalk-sportsbet/ds-abatar/tree/main" style="color: gray;">here.</a></p>',
    unsafe_allow_html=True)
    st.write("")
    st.write("")

    # ABatar logo
    left_co, cen_co, right_co = st.columns([2,1,2])
    with cen_co:
        st.image("abatarapp/gallery/references/abatar_logo.png", width=250)
        # st.image("abatarapp/gallery/references/powered_by_DS.png", width=260)
    
    st.write("")
    st.write("")
    # Second paragraph
    st.markdown(
        '<p class="custom-subheader" style="font-size: 25px; color: gray;">'
        'The backend engine of this app is <strong>ABatar</strong> - a python package powered by Sportsbet Data Science team.<br>'
        'Report bug or send general query and feature requests to the slack channel #abatar-abtest-analytics.</p>',
        unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Databricks Cluster button: start the cluster
    if st.button("Start Databricks Cluster", type="primary"):

        ws = WorkspaceClient()
        cluster_id = os.getenv("cluster_id")

        # Get the cluster's state
        while True:
            cluster_info = ws.clusters.get(cluster_id=cluster_id)
            cluster_name = cluster_info.cluster_name
            cluster_state = str(cluster_info.state).split('.')[-1]

            if cluster_state == 'TERMINATED':
                ws.clusters.start(cluster_id=cluster_id)
                st.write(f"Cluster {cluster_name} has been started.")
            else:
                st.write(f"Cluster {cluster_name} is in state:{cluster_state}.")

            if cluster_state == 'RUNNING':
                st.write(f"Cluster is :green[Active].")
                break

            time.sleep(90)
    
    st.session_state.demo = False
    if st.button("Activate Demo Mode", type="primary"):
        st.write(f"DEMO mode is :green[Active].")
        st.session_state.demo = True