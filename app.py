from __future__ import annotations

import streamlit as st

from src.app_backend import SustainabilitySchedulingBackend, resolve_default_paths
from ui.overview import render_overview_page
from ui.scheduler import render_scheduler_page
from utils.visualization import inject_app_styles


@st.cache_resource(show_spinner="Loading sustainability scheduling backend...")
def get_backend_resource(
    data_path: str,
    jobs_path: str,
    dc_config_path: str,
    coords_path: str,
) -> SustainabilitySchedulingBackend:
    return SustainabilitySchedulingBackend(
        data_path=data_path,
        jobs_path=jobs_path,
        dc_config_path=dc_config_path,
        coords_path=coords_path,
    )


def main() -> None:
    st.set_page_config(
        page_title="Sustainability-Aware Scheduler",
        page_icon=":material/alt_route:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_app_styles()

    defaults = resolve_default_paths()
    with st.sidebar:
        st.title("Scheduler App")
        page = st.radio("Page", ["Overview", "Scheduler Simulator"], label_visibility="visible")
        with st.expander("Data Sources", expanded=False):
            data_path = st.text_input("Master dataset", value=defaults["data_path"])
            jobs_path = st.text_input("Jobs dataset", value=defaults["jobs_path"])
            dc_config_path = st.text_input("DC config", value=defaults["dc_config_path"])
            coords_path = st.text_input("City coordinates", value=defaults["coords_path"])

    try:
        backend = get_backend_resource(data_path, jobs_path, dc_config_path, coords_path)
    except Exception as exc:
        st.error(f"Unable to initialize the scheduling backend: {exc}")
        return

    if page == "Overview":
        render_overview_page(backend)
    else:
        render_scheduler_page(backend)


if __name__ == "__main__":
    main()
