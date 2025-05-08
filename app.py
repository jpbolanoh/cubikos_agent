import streamlit as st
import pandas as pd
import sqlalchemy as sql
from langchain_openai import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define your MarketingDBAgent class
class MarketingDBAgent:
    """
    A specialized SQL agent for marketing database queries that handles token limits effectively.
    """
    
    def __init__(self, db_url, llm_model="gpt-4o-mini", temperature=0):
        """Initialize the agent with database connection and LLM"""
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))
        self.db = SQLDatabase.from_uri(db_url)
        
        # Get actual tables
       # Explicitly list your 18 tables here
        self.all_tables = [
            "brand",
            "brand_inte_dropi_account",
            "brand_inte_meta_account",
            "brand_inte_meta_account_adaccount",
            "brand_inte_meta_account_business",
            "brand_inte_meta_account_business_adaccount",
            "inte_dropi_account",
            "inte_dropi_account_product",
            "inte_dropi_order",
            "inte_dropi_order_product",
            "inte_dropi_product",
            "inte_meta_account",
            "inte_meta_adaccount",
            "inte_meta_business",
            "inte_meta_campaign",
            "inte_meta_campaign_insights",
            "integration",
            "integration_brand",   
        ]
        print(f"Using explicit list of {len(self.all_tables)} tables: {', '.join(self.all_tables)}")
        
        # Get JSONB column information
        self._get_jsonb_schema()
        
        # Initialize categories
        self.table_categories = {
            "ad_campaign": [],
            "product": []
        }
        
        # Categorize tables based on name patterns
        self._categorize_tables()
        
        # Create a reverse mapping from table to category
        self.table_to_category = {}
        for category, tables in self.table_categories.items():
            for table in tables:
                self.table_to_category[table] = category
        
        # Create the category extraction chain
        self._setup_chains()
    

    def _get_jsonb_schema(self):
        """Get schema information for JSONB columns only from the selected tables"""
        try:
            engine = sql.create_engine(self.db._engine.url)
            with engine.connect() as conn:
                # Store JSONB column information
                self.jsonb_info = {}
                
                # Only process the tables in our explicit list
                for table in self.all_tables:
                    # First check if the table has any JSONB columns
                    query = f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table}'
                    AND data_type = 'jsonb';
                    """
                    
                    jsonb_cols_df = pd.read_sql_query(sql.text(query), conn)
                    
                    # If this table has JSONB columns, get their keys
                    if not jsonb_cols_df.empty:
                        self.jsonb_info[table] = {}
                        
                        # For each JSONB column, get sample keys
                        for _, row in jsonb_cols_df.iterrows():
                            column = row['column_name']
                            
                            # Get sample keys from this JSONB column
                            jsonb_keys = get_jsonb_fields(conn, table, column)
                            
                            # Store keys for this column
                            self.jsonb_info[table][column] = jsonb_keys
                            
                            print(f"JSONB column found: {table}.{column} with {len(jsonb_keys)} keys")
        except Exception as e:
            print(f"Error getting JSONB schema: {e}")
            self.jsonb_info = {}
    
    def _categorize_tables(self):
        """
        Categorize tables into ad_campaign and product categories.
        """
        # Define patterns for the two categories
        ad_campaign_patterns = ["campaign", "ad", "meta", "ads", "marketing", "audience", "insight"]
        product_patterns = ["product", "item", "catalog", "inventory", "order", "dropi"]
        
        # Categorize each table
        for table in self.all_tables:
            table_lower = table.lower()
            
            # Check if table is related to ad campaigns
            if any(pattern in table_lower for pattern in ad_campaign_patterns):
                self.table_categories["ad_campaign"].append(table)
            
            # Check if table is related to products
            elif any(pattern in table_lower for pattern in product_patterns):
                self.table_categories["product"].append(table)
            
            # If no matches, add to the category with fewer tables (for balance)
            else:
                if len(self.table_categories["ad_campaign"]) <= len(self.table_categories["product"]):
                    self.table_categories["ad_campaign"].append(table)
                else:
                    self.table_categories["product"].append(table)
        
        # Debug output
        for category, tables in self.table_categories.items():
            print(f"Category '{category}' has {len(tables)} tables: {', '.join(tables[:5])}...")
    
    def _setup_chains(self):
        """Set up the LangChain processing chains"""
        # Create a system message to extract relevant categories
        category_system_message = """
        Return the categories of tables that are relevant to the user question.
        The available categories are: ad_campaign, product
        
        For each category, return true if it's relevant to the question, false otherwise.
        """
        
        # Create the function for category extraction
        extract_categories_function = {
            "name": "extract_categories",
            "description": "Extract relevant table categories from the question",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "object",
                        "properties": {
                            "ad_campaign": {
                                "type": "boolean", 
                                "description": "True if the question is about ad campaigns, ads, marketing"
                            },
                            "product": {
                                "type": "boolean", 
                                "description": "True if the question is about products, orders, inventory"
                            }
                        }
                    }
                },
                "required": ["categories"]
            }
        }
        
        # Create the extraction prompt
        category_prompt = ChatPromptTemplate.from_messages([
            ("system", category_system_message),
            ("human", "{question}")
        ])
        
        # Create the category extraction chain
        self.category_chain = category_prompt | self.llm.bind(functions=[extract_categories_function]) | JsonOutputFunctionsParser()
        
        # Function to get tables based on categories
        def get_tables_from_categories(categories_obj):
            relevant_tables = []
            for category, is_relevant in categories_obj["categories"].items():
                if is_relevant:
                    relevant_tables.extend(self.table_categories.get(category, []))
            
            # Ensure we have at least some tables to work with
            if not relevant_tables and self.all_tables:
                # If the question contains table names directly, use those
                return self.all_tables
                
            return list(set(relevant_tables))  # Remove duplicates
        
        # Complete chain to get tables
        self.table_chain = self.category_chain | get_tables_from_categories
        
        # Create the SQL query chain that will use only the relevant tables
        self.query_chain = create_sql_query_chain(self.llm, self.db)
        
        # Assign tables to use from table_chain result
        self.full_chain = RunnablePassthrough.assign(
            table_names_to_use={"question": itemgetter("question")} | self.table_chain
        ) | self.query_chain
    
    def execute_query(self, user_question):
        """Process a user question and execute the appropriate SQL query"""
        print(f"Processing question: {user_question}")
        
        try:
            # First, get the relevant tables
            start_time = time.time()
            relevant_tables = self.table_chain.invoke({"question": user_question})
            category_time = time.time() - start_time
            print(f"Identified {len(relevant_tables)} relevant tables in {category_time:.2f}s: {', '.join(relevant_tables)}...")
            
            if not relevant_tables:
                return {
                    "success": False,
                    "error": "No relevant tables found for this question."
                }
            
            # Get SQL query from the chain
            start_time = time.time()
            
            # If we have JSONB columns in the relevant tables, add this information to the prompt
            jsonb_info_text = ""
            for table in relevant_tables:
                if table in self.jsonb_info:
                    for column, keys in self.jsonb_info[table].items():
                        if keys:
                            jsonb_info_text += f"\nTable '{table}' has JSONB column '{column}' with keys: {', '.join(keys[:15])}"
                            jsonb_info_text += f"\nTo access these keys use: {column}->>'key_name'"
            
            # Add JSONB info to the question if available
            enhanced_question = user_question
            if jsonb_info_text:
                enhanced_question = f"{user_question}\n\nIMPORTANT JSONB INFORMATION: {jsonb_info_text}"
                print(f"Added JSONB info to question: {jsonb_info_text}")
            
            # Now invoke the chain with the enhanced question
            sql_query_raw = self.full_chain.invoke({"question": enhanced_question})
            query_gen_time = time.time() - start_time
            
            # Clean up the SQL query - remove markdown formatting
            sql_query = sql_query_raw.strip()
            
            # Remove markdown code block indicators (```sql and ```)
            if sql_query.startswith("```"):
                # Find the end of the first line
                first_line_end = sql_query.find("\n")
                if first_line_end != -1:
                    # Remove the first line (```sql)
                    sql_query = sql_query[first_line_end+1:]
                
                # Remove trailing backticks
                if sql_query.endswith("```"):
                    sql_query = sql_query[:-3].strip()
            
            print(f"Generated SQL in {query_gen_time:.2f}s: {sql_query}")
            
            # Add a LIMIT if not present
            if "LIMIT" not in sql_query.upper():
                sql_query = sql_query.rstrip(';') + " LIMIT 100;"
            
            # Execute the query
            try:
                start_time = time.time()
                
                # Create a fresh connection
                engine = sql.create_engine(self.db._engine.url)
                with engine.connect() as conn:
                    from sqlalchemy import text
                    df = pd.read_sql(text(sql_query), conn)
                
                execution_time = time.time() - start_time
                print(f"Query executed in {execution_time:.2f}s, returned {len(df)} rows")
                
                # Generate analysis of the result
                if not df.empty:
                    analysis_prompt = f"""
                    Eres un Experto en Análisis de Marketing. Analiza estos datos brevemente:
                    
                    Pregunta: {user_question}
                    
                    Forma de los datos: {df.shape[0]} filas, {df.shape[1]} columnas
                    Columnas: {', '.join(df.columns.tolist())}
                    Muestra (primeras 5 filas):
                    {df.head(5).to_string()}
                    
                    Proporciona un breve análisis (3-5 oraciones) de lo que muestran estos datos en relación con la pregunta.
                    """
                    
                    analysis_response = self.llm.invoke(analysis_prompt)
                    analysis = analysis_response.content
                else:
                    analysis = "The query returned no data."
                
                return {
                    "success": True,
                    "sql_query": sql_query,
                    "data": df,
                    "analysis": analysis,
                    "relevant_tables": relevant_tables,
                    "timing": {
                        "category_identification": category_time,
                        "query_generation": query_gen_time,
                        "query_execution": execution_time
                    }
                }
            except Exception as e:
                print(f"Error executing query: {e}")
                
                # Try to fix the query
                fix_prompt = f"""
                The SQL query failed with error: {str(e)}
                
                Original query:
                {sql_query}
                
                This is for PostgreSQL database. 
                Please provide a fixed version of this query that will run successfully.
                Important: Return ONLY the SQL code itself without any markdown formatting or backticks.
                For PostgreSQL, make sure to use double quotes for column names with capital letters.
                You can use 'public.' prefix for table names.
                
                JSONB Information:
                {jsonb_info_text}
                """
                
                try:
                    fix_response = self.llm.invoke(fix_prompt)
                    fixed_sql = fix_response.content.strip()
                    
                    # Clean up the fixed SQL - remove any markdown formatting
                    if fixed_sql.startswith("```"):
                        # Find the end of the first line
                        first_line_end = fixed_sql.find("\n")
                        if first_line_end != -1:
                            # Remove the first line (```sql)
                            fixed_sql = fixed_sql[first_line_end+1:]
                        
                        # Remove trailing backticks
                        if fixed_sql.endswith("```"):
                            fixed_sql = fixed_sql[:-3].strip()
                    
                    print(f"Attempting fixed query: {fixed_sql}")
                    
                    # Add a LIMIT if not present
                    if "LIMIT" not in fixed_sql.upper():
                        fixed_sql = fixed_sql.rstrip(';') + " LIMIT 100;"
                    
                    # Try the fixed query
                    with engine.connect() as conn:
                        df = pd.read_sql(text(fixed_sql), conn)
                    if not df.empty:
                        analysis_prompt = f"""
                        Eres un Experto en Análisis de Marketing. Analiza estos datos brevemente:
                        
                        Pregunta: {user_question}
                        
                        Forma de los datos: {df.shape[0]} filas, {df.shape[1]} columnas
                        Columnas: {', '.join(df.columns.tolist())}
                        Muestra (primeras 5 filas):
                        {df.head(5).to_string()}
                        
                        Proporciona un breve análisis (3-5 oraciones) de lo que muestran estos datos en relación con la pregunta.
                        """
                        
                        analysis_response = self.llm.invoke(analysis_prompt)
                        analysis = analysis_response.content
                    else:
                        analysis = "La consulta no devolvió datos."
                    
                    return {
                        "success": True,
                        "sql_query": fixed_sql,
                        "original_query": sql_query,
                        "data": df,
                        "analysis": analysis,
                        "relevant_tables": relevant_tables
                    }
                except Exception as e2:
                    return {
                        "success": False,
                        "sql_query": sql_query,
                        "fixed_sql": fixed_sql if 'fixed_sql' in locals() else None,
                        "error": f"Original error: {str(e)}\nError with fixed query: {str(e2)}",
                        "relevant_tables": relevant_tables
                    }
        except Exception as e:
            print(f"Error in query process: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

# Function to extract JSONB keys
def get_jsonb_fields(conn, table_name, jsonb_column):
    try:
        query = f"""
        SELECT DISTINCT jsonb_object_keys({jsonb_column}) AS key_name
        FROM {table_name}
        LIMIT 100;
        """
        keys_df = pd.read_sql(query, conn)
        if not keys_df.empty:
            return keys_df['key_name'].tolist()
        else:
            return []
    except Exception as e:
        print(f"Error getting JSONB keys: {e}")
        return []

# Streamlit App Configuration
st.set_page_config(page_title="Marketing Database Query Tool", page_icon="📊")
st.title("Marketing Database Query Tool")

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
        conn = sql_engine.connect()
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

# LLM setup
with st.spinner("Initializing language model..."):
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY
        )
        st.sidebar.success("✅ LLM initialized")
    except Exception as e:
        st.error(f"Error initializing language model: {e}")
        st.stop()

# Initialize agents
agent_type = st.sidebar.radio(
    "Select Query Agent Type",
    ["LangChain Agent (for complex DBs)", "Direct SQL"]
)

if agent_type == "LangChain Agent (for complex DBs)":
    with st.spinner("Initializing LangChain SQL Agent..."):
        try:
            db_agent = MarketingDBAgent(
                db_url=DB_URL,
                llm_model="gpt-4o-mini"
            )
            st.sidebar.success("✅ LangChain SQL Agent initialized")
        except Exception as e:
            st.error(f"Error initializing LangChain SQL Agent: {e}")
            st.stop()
else:
    st.sidebar.success("✅ Using Direct SQL")

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Main app content
st.markdown("""
This tool allows you to query your marketing database using natural language. Simply type your question
and the AI will translate it to SQL and return the results.
""")

with st.expander("Example Questions", expanded=False):
    st.markdown("""
    - Muéstrame 5 filas de inte_dropi_order
    - Cuáles son las 10 campañas con mayor gasto este mes?
    - Análisis de rendimiento de campañas por día
    - Cuál es el CPC promedio por campaña?
    """)

# Query input
question = st.text_area("Enter your question", height=100)

if st.button("Submit Query"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Processing your question..."):
            start_time = time.time()
            
            if agent_type == "LangChain Agent (for complex DBs)":
                # Use the LangChain agent
                result = db_agent.execute_query(question)
                
                # Display results
                if result["success"]:
                    st.success(f"Query processed in {time.time() - start_time:.2f} seconds")
                    
                    with st.expander("SQL Query", expanded=True):
                        st.code(result["sql_query"], language="sql")
                    
                    if "relevant_tables" in result:
                        st.caption(f"Relevant tables: {', '.join(result['relevant_tables'][:5])}")
                    
                    if "data" in result and isinstance(result["data"], pd.DataFrame):
                        st.subheader("Results")
                        st.dataframe(result["data"])
                        
                        # Add download button
                        csv = result["data"].to_csv(index=False)
                        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv",
                        )
                    
                    if "analysis" in result:
                        st.subheader("Analysis")
                        st.write(result["analysis"])
                    
                    if debug_mode and "timing" in result:
                        st.subheader("Performance Metrics")
                        st.json(result["timing"])
                else:
                    st.error("Query failed")
                    st.error(result["error"])
                    
                    if "sql_query" in result:
                        with st.expander("Generated SQL (with error)"):
                            st.code(result["sql_query"], language="sql")
                    
                    if "fixed_sql" in result and result["fixed_sql"]:
                        st.write("I attempted to fix the query, but it still failed:")
                        st.code(result["fixed_sql"], language="sql")
            else:
                # Direct SQL execution
                try:
                    # First, use LLM to convert question to SQL
                    sql_prompt = f"""
                    Translate this question to PostgreSQL syntax: "{question}"
                    Include LIMIT 100 for safety.
                    Return ONLY the SQL code with no explanations or markdown.
                    """
                    
                    sql_response = llm.invoke(sql_prompt)
                    sql_query = sql_response.content.strip()
                    
                    # Display generated SQL
                    with st.expander("Generated SQL", expanded=True):
                        st.code(sql_query, language="sql")
                    
                    # Execute the query
                    df = pd.read_sql_query(sql.text(sql_query), conn)
                    
                    # Show results
                    st.subheader("Results")
                    st.dataframe(df)
                    
                    # Add download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv",
                    )
                    
                    # Generate analysis
                    analysis_prompt = f"""
                    As a Marketing Analytics Expert, provide a brief analysis of this data:
                    
                    Question: {question}
                    Columns: {', '.join(df.columns.tolist())}
                    Row count: {len(df)}
                    
                    Give 3-5 sentences of analysis and insights.
                    """
                    
                    analysis_response = llm.invoke(analysis_prompt)
                    
                    st.subheader("Analysis")
                    st.write(analysis_response.content)
                    
                except Exception as e:
                    st.error(f"Error executing query: {e}")

# Database Explorer
with st.expander("Database Explorer", expanded=False):
    st.subheader("Database Explorer")
    
    try:
        # Get table list
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        tables_df = pd.read_sql_query(tables_query, conn)
        
        # Allow filtering by category
        category_filter = st.radio(
            "Filter by category",
            ["All", "Ad Campaign", "Product"]
        )
        
        if category_filter != "All" and agent_type == "LangChain Agent (for complex DBs)":
            category_key = category_filter.lower().replace(" ", "_")
            filtered_tables = db_agent.table_categories.get(category_key, [])
            filtered_df = tables_df[tables_df['table_name'].isin(filtered_tables)]
            tables_df = filtered_df
        
        selected_table = st.selectbox("Select a table to explore", tables_df['table_name'].tolist())
        
        if selected_table:
            # Get columns
            columns_query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{selected_table}'
            ORDER BY ordinal_position;
            """
            columns_df = pd.read_sql_query(columns_query, conn)
            
            st.write("Table columns:")
            st.dataframe(columns_df)
            
            # Check for JSONB columns
            jsonb_columns = columns_df[columns_df['data_type'] == 'jsonb']['column_name'].tolist()
            if jsonb_columns:
                selected_jsonb = st.selectbox("Explore JSONB column", jsonb_columns)
                if selected_jsonb:
                    jsonb_keys = get_jsonb_fields(conn, selected_table, selected_jsonb)
                    st.write(f"Keys in JSONB column '{selected_jsonb}':")
                    st.write(", ".join(jsonb_keys[:50]))
            
            # Show sample data
            if st.button("Show sample data"):
                sample_query = f"SELECT * FROM {selected_table} LIMIT 5"
                sample_df = pd.read_sql_query(sample_query, conn)
                st.dataframe(sample_df)
    except Exception as e:
        st.error(f"Error exploring database: {e}")

# Direct SQL query tool
with st.expander("Direct SQL Query Tool", expanded=False):
    st.subheader("Run SQL Query Directly")
    
    direct_sql = st.text_area("Enter SQL Query", height=150)
    
    if st.button("Execute SQL"):
        if not direct_sql:
            st.warning("Please enter a SQL query")
        else:
            try:
                with st.spinner("Executing query..."):
                    start_time = time.time()
                    result_df = pd.read_sql_query(sql.text(direct_sql), conn)
                    
                st.success(f"Query executed in {time.time() - start_time:.2f} seconds")
                st.write(f"Returned {len(result_df)} rows")
                st.dataframe(result_df)
                
                # Add download button
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="direct_query_results.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error executing SQL: {e}")

# Footer
st.markdown("---")
st.caption("Marketing Database Query Tool | Simplified with ad_campaign and product categories")