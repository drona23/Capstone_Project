from __future__ import annotations

import streamlit as st

from src.dashboard.overview_page import render_overview_page
from src.dashboard.scheduler_page import render_scheduler_page


st.set_page_config(
    page_title="Making AI Less Thirsty",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page(render_overview_page, title="Overview", icon=":material/analytics:", default=True),
        st.Page(render_scheduler_page, title="Scheduler Simulator", icon=":material/alt_route:"),
    ]
)
navigation.run()
