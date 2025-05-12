import streamlit as st
import pandas as pd
import sqlalchemy as sql
import os
from dotenv import load_dotenv

# Import UI components
from src.ui.general_query import show_general_query_tab
from src.ui.correlation import show_correlation_tab
from src.ui.explorer import show_explorer_tab
from src.ui.direct_sql import show_direct_sql_tab

# Import agent
from src.agents.marketing_db_agent import MarketingDBAgent

# Import utilities
from src.utils.db_utils import test_connection

# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(page_title="Marketing Database Query Tool", page_icon="📊", layout="wide")
st.title("Marketing Database Query Tool")

# Initialize session state for shared components
if 'db_agent' not in st.session_state:
    st.session_state.db_agent = None
if 'conn' not in st.session_state:
    st.session_state.conn = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Sidebar configuration
st.sidebar.header("Configuration")

# Database Connection
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    st.error("Database URL not found. Please check your .env file.")
    st.stop()

# Connect to database
try:
    with st.spinner("Connecting to database..."):
        sql_engine = sql.create_engine(DB_URL)
        st.session_state.conn = sql_engine.connect()
    st.sidebar.success("✅ Database connected")
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if not OPENAI_API_KEY:
        st.error("Please enter your OpenAI API Key")
        st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize agent if not already initialized
if st.session_state.db_agent is None:
    with st.spinner("Initializing Marketing DB Agent..."):
        try:
            st.session_state.db_agent = MarketingDBAgent(
                db_url=DB_URL,
                llm_model="gpt-4o-mini"
            )
            st.sidebar.success("✅ Marketing DB Agent initialized")
        except Exception as e:
            st.error(f"Error initializing Marketing DB Agent: {e}")
            st.stop()

# Debug mode toggle
st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    "Natural Language Query", 
    "Product-Campaign Correlation", 
    "Database Explorer", 
    "Direct SQL"
])

# Populate each tab
with tab1:
    show_general_query_tab(st.session_state.db_agent, st.session_state.debug_mode)

with tab2:
    show_correlation_tab(st.session_state.db_agent, st.session_state.conn)

with tab3:
    show_explorer_tab(st.session_state.conn, st.session_state.db_agent)

with tab4:
    show_direct_sql_tab(st.session_state.conn)

# Footer
st.markdown("---")
st.caption("Marketing Database Query Tool | Enhanced with Product-Campaign Correlation Analysis")