from __future__ import annotations

import streamlit as st

from ui.scheduler import render_scheduler_page
from utils.visualization import inject_app_styles


def main() -> None:
    st.set_page_config(
        page_title="Sustainability Routing Simulation",
        page_icon=":material/alt_route:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_app_styles()
    render_scheduler_page()


if __name__ == "__main__":
    main()
