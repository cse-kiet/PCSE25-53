import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st

from app import ui
from app import resources

st.set_page_config(page_title="Travel Planner", layout="wide")

# Load global resources. This call ensures that the resources are loaded and stored within the app.resources module.
resources.load_resources()

# Initialize session state (uses functions now in app.ui)
ui.initialize_session_state()

# ----------------------------
# Main Application with Tabs
# ----------------------------
tab1_title = "🔍 Image Search"
tab2_title = "🗺️ Place Discovery"
tab3_title = "📅 Itinerary Planner"

tab1, tab2, tab3 = st.tabs([tab1_title, tab2_title, tab3_title])

with tab1:
    ui.render_tab1()
with tab2:
    ui.render_tab2()
with tab3:
    ui.render_tab3()
