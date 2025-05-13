from langchain_openai import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
import time
import pandas as pd
import sqlalchemy as sql
import os

from src.utils.db_utils import get_jsonb_fields

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
        
    
    def _convert_uuids_to_string(self, df):
        """Helper method to convert UUID columns to strings to avoid PyArrow errors"""
        if df is None or df.empty:
            return
            
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if column contains UUID objects
                if df[column].notnull().any() and isinstance(df[column].iloc[0], object):
                    try:
                        if hasattr(df[column].iloc[0], 'hex'):  # Check if it's UUID-like
                            df[column] = df[column].astype(str)
                    except:
                        pass

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
            query_with_context = user_question
            if jsonb_info_text:
                query_with_context = f"{user_question}\n\nIMPORTANT JSONB INFORMATION: {jsonb_info_text}"
                print(f"Added JSONB info to question: {jsonb_info_text}")
            
            # Now invoke the chain with the enhanced question
            sql_query_raw = self.full_chain.invoke({"question": query_with_context})
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
                    
                    # Convert UUID columns to strings to avoid PyArrow errors
                    for column in df.columns:
                        if df[column].dtype == 'object':
                            # Check if column contains UUID objects
                            if df[column].notnull().any() and isinstance(df[column].iloc[0], object):
                                try:
                                    if hasattr(df[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                        df[column] = df[column].astype(str)
                                except:
                                    pass
                
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

    def get_products(self, limit=100):
        """
        Retrieve a list of products from the database
        """
        try:
            # Look for product-related tables
            product_tables = [t for t in self.all_tables if "product" in t.lower()]
            
            if not product_tables:
                return pd.DataFrame()
            
            # Find the main product table
            main_product_table = next((t for t in product_tables if t == "inte_dropi_product"), product_tables[0])
            
            # Create a SQL query to get products
            sql_query = f"""
            SELECT DISTINCT p.* 
            FROM {main_product_table} p
            INNER JOIN inte_dropi_order_product op ON p.id = op."productId"
            LIMIT {limit}
            """
            
            # Execute the query
            engine = sql.create_engine(self.db._engine.url)
            with engine.connect() as conn:
                from sqlalchemy import text
                df = pd.read_sql(text(sql_query), conn)
                
                # Convert UUID columns to strings to avoid PyArrow errors
                for column in df.columns:
                    if df[column].dtype == 'object':
                        # Check if column contains UUID objects
                        if df[column].notnull().any() and isinstance(df[column].iloc[0], object):
                            try:
                                if hasattr(df[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                    df[column] = df[column].astype(str)
                            except:
                                pass
            
            return df
        except Exception as e:
            print(f"Error getting products: {e}")
            return pd.DataFrame()
    
    def get_campaigns(self, limit=100):
        """
        Retrieve a list of campaigns from the database
        """
        try:
            # Look for campaign-related tables
            campaign_tables = [t for t in self.all_tables if "campaign" in t.lower()]
            
            if not campaign_tables:
                return pd.DataFrame()
            
            # Find the main campaign table
            main_campaign_table = next((t for t in campaign_tables if t == "inte_meta_campaign"), campaign_tables[0])
            
            # Create a SQL query to get campaigns
            sql_query = f"""
            SELECT DISTINCT c.* 
            FROM {main_campaign_table} c
            INNER JOIN inte_meta_campaign_insights ci ON c.id = ci."parentId"
            LIMIT {limit}
            """
            # Execute the query
            engine = sql.create_engine(self.db._engine.url)
            with engine.connect() as conn:
                from sqlalchemy import text
                df = pd.read_sql(text(sql_query), conn)
                
                # Convert UUID columns to strings to avoid PyArrow errors
                for column in df.columns:
                    if df[column].dtype == 'object':
                        # Check if column contains UUID objects
                        if df[column].notnull().any() and isinstance(df[column].iloc[0], object):
                            try:
                                if hasattr(df[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                    df[column] = df[column].astype(str)
                            except:
                                pass
            
            return df
        except Exception as e:
            print(f"Error getting campaigns: {e}")
            return pd.DataFrame()
    
    def execute_correlation_query(self, product_id, campaign_id, user_question=None):
        """
        Analyze how a marketing campaign may have influenced product performance.
        This uses the LLM to generate appropriate queries based on the actual schema.
        
        Parameters:
        -----------
        product_id : str
            The ID of the product to analyze
        campaign_id : str
            The ID of the campaign to analyze
        user_question : str, optional
            A specific question from the user that should be addressed in the analysis
            
        Returns:
        --------
        dict
            A dictionary containing campaign insights, product order data, AI analysis and answer to user question
        """
        try:
            print(f"Analyzing impact of campaign ID {campaign_id} on product ID {product_id}")
            if user_question:
                print(f"User question: {user_question}")
            
            # Start timing
            start_time = time.time()
            
            # Find relevant tables
            insights_table = next((t for t in self.all_tables if t == "inte_meta_campaign_insights"), None)
            product_table = next((t for t in self.all_tables if t == "inte_dropi_product"), None)
            orders_table = next((t for t in self.all_tables if t == "inte_dropi_order"), None)
            order_product_table = next((t for t in self.all_tables if t == "inte_dropi_order_product"), None)
            
            if not insights_table or not product_table or not orders_table:
                return {
                    "success": False,
                    "error": "Required tables not found in database (inte_meta_campaign_insights, inte_dropi_product, or inte_dropi_order)"
                }
            
            # Connect to database
            engine = sql.create_engine(self.db._engine.url)
            with engine.connect() as conn:
                from sqlalchemy import text
                
                # Get table schema information
                def get_table_columns(table_name):
                    query = f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position;
                    """
                    return pd.read_sql(text(query), conn)['column_name'].tolist()
                
                insights_columns = get_table_columns(insights_table)
                product_columns = get_table_columns(product_table)
                orders_columns = get_table_columns(orders_table)
                order_product_columns = get_table_columns(order_product_table) if order_product_table else []
                
                print(f"Campaign insights columns: {insights_columns}")
                print(f"Product columns: {product_columns}")
                print(f"Orders columns: {orders_columns}")
                if order_product_table:
                    print(f"Order-product columns: {order_product_columns}")
                
                # Get JSONB column information for insights table
                jsonb_info_text = ""
                if insights_table in self.jsonb_info:
                    for column, keys in self.jsonb_info[insights_table].items():
                        if keys:
                            jsonb_info_text += f"\nTable '{insights_table}' has JSONB column '{column}' with keys: {', '.join(keys[:15])}"
                            jsonb_info_text += f"\nTo access these keys use: {column}->>'key_name'"
                    print(f"JSONB info: {jsonb_info_text}")
                
                #--------------------------------------------------------------------
                # 1. Get campaign insights data
                #--------------------------------------------------------------------
                insights_prompt = f"""
                Generate a PostgreSQL query to get campaign insights data.
                
                TASK:
                Retrieve campaign insights for campaign ID '{campaign_id}'
                
                TABLES AND COLUMNS:
                - Table '{insights_table}' has columns: {', '.join(insights_columns)}
                
                REQUIREMENTS:
                - Find insights where the campaign ID column (likely called id, campaignId, or "parentId") equals '{campaign_id}'
                - Order by date/time column (if exists) in descending order
                - Limit to 20 rows
                - Only include working column names that exist in the table
                - Remember to use double quotes for column names if they contain capital letters or special characters
                {jsonb_info_text}
                
                Important: Return ONLY the SQL code itself without any explanation, markdown formatting or backticks.
                """
                
                insights_response = self.llm.invoke(insights_prompt)
                campaign_insights_query = insights_response.content.strip()
                
                # Clean up the query
                if campaign_insights_query.startswith("```"):
                    first_line_end = campaign_insights_query.find("\n")
                    if first_line_end != -1:
                        campaign_insights_query = campaign_insights_query[first_line_end+1:]
                    if campaign_insights_query.endswith("```"):
                        campaign_insights_query = campaign_insights_query[:-3].strip()
                
                # Add a LIMIT if not present
                if "LIMIT" not in campaign_insights_query.upper():
                    campaign_insights_query = campaign_insights_query.rstrip(';') + " LIMIT 20;"
                
                print(f"Generated campaign insights query: {campaign_insights_query}")
                
                # Execute the insights query
                try:
                    df_insights = pd.read_sql(text(campaign_insights_query), conn)
                    if df_insights.empty:
                        print(f"No insights found for campaign ID {campaign_id}")
                    else:
                        print(f"Found campaign insights: {len(df_insights)} rows")
                        
                    # Convert UUID columns to strings
                    for column in df_insights.columns:
                        if df_insights[column].dtype == 'object':
                            if df_insights[column].notnull().any() and isinstance(df_insights[column].iloc[0], object):
                                try:
                                    if hasattr(df_insights[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                        df_insights[column] = df_insights[column].astype(str)
                                except:
                                    pass
                    
                except Exception as e_insights:
                    print(f"Error retrieving campaign insights: {e_insights}")
                    
                    # Try to fix the query
                    fix_insights_prompt = f"""
                    The SQL query failed with error: {str(e_insights)}
                    
                    Original query:
                    {campaign_insights_query}
                    
                    This is for PostgreSQL database. 
                    Please provide a fixed version of this query that will run successfully.
                    
                    TABLES AND COLUMNS:
                    - Table '{insights_table}' has columns: {', '.join(insights_columns)}
                    
                    Important: Return ONLY the SQL code itself without any markdown formatting or backticks.
                    For PostgreSQL, make sure to use double quotes for column names with capital letters or special characters.
                    """
                    
                    try:
                        fix_response = self.llm.invoke(fix_insights_prompt)
                        fixed_insights_query = fix_response.content.strip()
                        
                        # Clean up the fixed query
                        if fixed_insights_query.startswith("```"):
                            first_line_end = fixed_insights_query.find("\n")
                            if first_line_end != -1:
                                fixed_insights_query = fixed_insights_query[first_line_end+1:]
                            if fixed_insights_query.endswith("```"):
                                fixed_insights_query = fixed_insights_query[:-3].strip()
                        
                        print(f"Attempting fixed insights query: {fixed_insights_query}")
                        
                        # Add a LIMIT if not present
                        if "LIMIT" not in fixed_insights_query.upper():
                            fixed_insights_query = fixed_insights_query.rstrip(';') + " LIMIT 20;"
                        
                        df_insights = pd.read_sql(text(fixed_insights_query), conn)
                        
                        # Convert UUID columns to strings
                        for column in df_insights.columns:
                            if df_insights[column].dtype == 'object':
                                if df_insights[column].notnull().any() and isinstance(df_insights[column].iloc[0], object):
                                    try:
                                        if hasattr(df_insights[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                            df_insights[column] = df_insights[column].astype(str)
                                    except:
                                        pass
                        
                        campaign_insights_query = fixed_insights_query
                        
                    except Exception as e2:
                        print(f"Error with fixed insights query: {e2}")
                        df_insights = pd.DataFrame()
                
                #--------------------------------------------------------------------
                # 2. Get product information
                #--------------------------------------------------------------------
                product_prompt = f"""
                Generate a PostgreSQL query to get product information.
                
                TASK:
                Retrieve product data for product ID '{product_id}'
                
                TABLES AND COLUMNS:
                - Table '{product_table}' has columns: {', '.join(product_columns)}
                
                REQUIREMENTS:
                - Find the product where the product ID column (likely called id or productId) equals '{product_id}'
                - Only include working column names that exist in the table
                - Remember to use double quotes for column names if they contain capital letters or special characters
                - Limit to 1 row
                
                Important: Return ONLY the SQL code itself without any explanation, markdown formatting or backticks.
                """
                
                product_response = self.llm.invoke(product_prompt)
                product_query = product_response.content.strip()
                
                # Clean up the query
                if product_query.startswith("```"):
                    first_line_end = product_query.find("\n")
                    if first_line_end != -1:
                        product_query = product_query[first_line_end+1:]
                    if product_query.endswith("```"):
                        product_query = product_query[:-3].strip()
                
                # Add a LIMIT if not present
                if "LIMIT" not in product_query.upper():
                    product_query = product_query.rstrip(';') + " LIMIT 1;"
                
                print(f"Generated product query: {product_query}")
                
                # Execute the product query
                try:
                    df_product = pd.read_sql(text(product_query), conn)
                    if df_product.empty:
                        print(f"No product found with ID {product_id}")
                    else:
                        print(f"Found product data: {len(df_product)} rows")
                        
                    # Convert UUID columns to strings
                    for column in df_product.columns:
                        if df_product[column].dtype == 'object':
                            if df_product[column].notnull().any() and isinstance(df_product[column].iloc[0], object):
                                try:
                                    if hasattr(df_product[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                        df_product[column] = df_product[column].astype(str)
                                except:
                                    pass
                    
                except Exception as e_product:
                    print(f"Error retrieving product: {e_product}")
                    
                    # Try to fix the query
                    fix_product_prompt = f"""
                    The SQL query failed with error: {str(e_product)}
                    
                    Original query:
                    {product_query}
                    
                    This is for PostgreSQL database. 
                    Please provide a fixed version of this query that will run successfully.
                    
                    TABLES AND COLUMNS:
                    - Table '{product_table}' has columns: {', '.join(product_columns)}
                    
                    Important: Return ONLY the SQL code itself without any markdown formatting or backticks.
                    For PostgreSQL, make sure to use double quotes for column names with capital letters or special characters.
                    """
                    
                    try:
                        fix_response = self.llm.invoke(fix_product_prompt)
                        fixed_product_query = fix_response.content.strip()
                        
                        # Clean up the fixed query
                        if fixed_product_query.startswith("```"):
                            first_line_end = fixed_product_query.find("\n")
                            if first_line_end != -1:
                                fixed_product_query = fixed_product_query[first_line_end+1:]
                            if fixed_product_query.endswith("```"):
                                fixed_product_query = fixed_product_query[:-3].strip()
                        
                        print(f"Attempting fixed product query: {fixed_product_query}")
                        
                        # Add a LIMIT if not present
                        if "LIMIT" not in fixed_product_query.upper():
                            fixed_product_query = fixed_product_query.rstrip(';') + " LIMIT 1;"
                        
                        df_product = pd.read_sql(text(fixed_product_query), conn)
                        
                        # Convert UUID columns to strings
                        for column in df_product.columns:
                            if df_product[column].dtype == 'object':
                                if df_product[column].notnull().any() and isinstance(df_product[column].iloc[0], object):
                                    try:
                                        if hasattr(df_product[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                            df_product[column] = df_product[column].astype(str)
                                    except:
                                        pass
                        
                        product_query = fixed_product_query
                        
                    except Exception as e2:
                        print(f"Error with fixed product query: {e2}")
                        df_product = pd.DataFrame()
                
                #--------------------------------------------------------------------
                # 3. Get order data for the product
                #--------------------------------------------------------------------
                orders_prompt = f"""
                Generate a PostgreSQL query to get orders for a specific product.
                
                TASK:
                Retrieve orders related to product ID '{product_id}'
                
                TABLES AND COLUMNS:
                - Orders table '{orders_table}' has columns: {', '.join(orders_columns)}
                {f"- Order-product junction table '{order_product_table}' has columns: {', '.join(order_product_columns)}" if order_product_table else ""}
                
                REQUIREMENTS:
                - Get orders related to product ID '{product_id}'
                {"- Use the junction table to link orders and products" if order_product_table else "- Try to find a direct reference to product ID in the orders table"}
                - Order by date/time column (if exists) in descending order
                - Only include working column names that exist in the tables
                - Remember to use double quotes for column names if they contain capital letters or special characters
                - Limit to 30 results
                
                Important: Return ONLY the SQL code itself without any explanation, markdown formatting or backticks.
                """
                
                orders_response = self.llm.invoke(orders_prompt)
                orders_query = orders_response.content.strip()
                
                # Clean up the query
                if orders_query.startswith("```"):
                    first_line_end = orders_query.find("\n")
                    if first_line_end != -1:
                        orders_query = orders_query[first_line_end+1:]
                    if orders_query.endswith("```"):
                        orders_query = orders_query[:-3].strip()
                
                # Add a LIMIT if not present
                if "LIMIT" not in orders_query.upper():
                    orders_query = orders_query.rstrip(';') + " LIMIT 30;"
                
                print(f"Generated orders query: {orders_query}")
                
                # Execute the orders query
                try:
                    df_orders = pd.read_sql(text(orders_query), conn)
                    if df_orders.empty:
                        print(f"No orders found for product ID {product_id}")
                    else:
                        print(f"Found orders data: {len(df_orders)} rows")
                        
                    # Convert UUID columns to strings
                    for column in df_orders.columns:
                        if df_orders[column].dtype == 'object':
                            if df_orders[column].notnull().any() and isinstance(df_orders[column].iloc[0], object):
                                try:
                                    if hasattr(df_orders[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                        df_orders[column] = df_orders[column].astype(str)
                                except:
                                    pass
                    
                except Exception as e_orders:
                    print(f"Error retrieving orders: {e_orders}")
                    
                    # Try to fix the query
                    fix_orders_prompt = f"""
                    The SQL query failed with error: {str(e_orders)}
                    
                    Original query:
                    {orders_query}
                    
                    This is for PostgreSQL database. 
                    Please provide a fixed version of this query that will run successfully.
                    
                    TABLES AND COLUMNS:
                    - Orders table '{orders_table}' has columns: {', '.join(orders_columns)}
                    {f"- Order-product junction table '{order_product_table}' has columns: {', '.join(order_product_columns)}" if order_product_table else ""}
                    
                    Important: Return ONLY the SQL code itself without any explanation, markdown formatting or backticks.
                    For PostgreSQL, make sure to use double quotes for column names with capital letters or special characters.
                    """
                    
                    try:
                        fix_response = self.llm.invoke(fix_orders_prompt)
                        fixed_orders_query = fix_response.content.strip()
                        
                        # Clean up the fixed query
                        if fixed_orders_query.startswith("```"):
                            first_line_end = fixed_orders_query.find("\n")
                            if first_line_end != -1:
                                fixed_orders_query = fixed_orders_query[first_line_end+1:]
                            if fixed_orders_query.endswith("```"):
                                fixed_orders_query = fixed_orders_query[:-3].strip()
                        
                        print(f"Attempting fixed orders query: {fixed_orders_query}")
                        
                        # Add a LIMIT if not present
                        if "LIMIT" not in fixed_orders_query.upper():
                            fixed_orders_query = fixed_orders_query.rstrip(';') + " LIMIT 30;"
                        
                        df_orders = pd.read_sql(text(fixed_orders_query), conn)
                        
                        # Convert UUID columns to strings
                        for column in df_orders.columns:
                            if df_orders[column].dtype == 'object':
                                if df_orders[column].notnull().any() and isinstance(df_orders[column].iloc[0], object):
                                    try:
                                        if hasattr(df_orders[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                            df_orders[column] = df_orders[column].astype(str)
                                    except:
                                        pass
                        
                        orders_query = fixed_orders_query
                        
                    except Exception as e2:
                        print(f"Error with fixed orders query: {e2}")
                        df_orders = pd.DataFrame()
                
                #--------------------------------------------------------------------
                # 4. If we have campaign dates, get orders during campaign period
                #--------------------------------------------------------------------
                df_orders_during_campaign = pd.DataFrame()
                
                if not df_insights.empty and not df_orders.empty:
                    # Try to determine campaign date range
                    try:
                        # Find potential date columns in insights table
                        date_columns_insights = [col for col in df_insights.columns if any(term in col.lower() for term in ['date', 'time', 'created'])]
                        
                        if date_columns_insights:
                            # Use the first date column found
                            date_column_insights = date_columns_insights[0]
                            print(f"Using date column for insights: {date_column_insights}")
                            
                            try:
                                campaign_dates = df_insights[date_column_insights].sort_values()
                                start_date = campaign_dates.min()
                                end_date = campaign_dates.max()
                                
                                # Find potential date columns in orders table
                                date_columns_orders = [col for col in df_orders.columns if any(term in col.lower() for term in ['date', 'time', 'created'])]
                                
                                if date_columns_orders:
                                    # Use the first date column found
                                    date_column_orders = date_columns_orders[0]
                                    print(f"Using date column for orders: {date_column_orders}")
                                    
                                    print(f"Campaign period: {start_date} to {end_date}")
                                    
                                    # Generate a query for orders during campaign period
                                    campaign_orders_prompt = f"""
                                    Generate a PostgreSQL query to get orders for a product during a specific date range.
                                    
                                    TASK:
                                    Retrieve orders for product ID '{product_id}' during the period from '{start_date}' to '{end_date}'
                                    
                                    TABLES AND COLUMNS:
                                    - Orders table '{orders_table}' has columns: {', '.join(orders_columns)}
                                    {f"- Order-product junction table '{order_product_table}' has columns: {', '.join(order_product_columns)}" if order_product_table else ""}
                                    
                                    REQUIREMENTS:
                                    - Get orders related to product ID '{product_id}'
                                    - Filter orders between '{start_date}' and '{end_date}' using the date column '{date_column_orders}'
                                    {"- Use the junction table to link orders and products" if order_product_table else ""}
                                    - Only include working column names that exist in the tables
                                    - Remember to use double quotes for column names if they contain capital letters or special characters
                                    - Order by date column
                                    - Limit to 30 results
                                    
                                    Important: Return ONLY the SQL code itself without any explanation, markdown formatting or backticks.
                                    """
                                    
                                    campaign_orders_response = self.llm.invoke(campaign_orders_prompt)
                                    campaign_orders_query = campaign_orders_response.content.strip()
                                    
                                    # Clean up the query
                                    if campaign_orders_query.startswith("```"):
                                        first_line_end = campaign_orders_query.find("\n")
                                        if first_line_end != -1:
                                            campaign_orders_query = campaign_orders_query[first_line_end+1:]
                                        if campaign_orders_query.endswith("```"):
                                            campaign_orders_query = campaign_orders_query[:-3].strip()
                                    
                                    # Add a LIMIT if not present
                                    if "LIMIT" not in campaign_orders_query.upper():
                                        campaign_orders_query = campaign_orders_query.rstrip(';') + " LIMIT 30;"
                                    
                                    print(f"Generated campaign period orders query: {campaign_orders_query}")
                                    
                                    try:
                                        df_orders_during_campaign = pd.read_sql(text(campaign_orders_query), conn)
                                        
                                        # Convert UUID columns to strings
                                        for column in df_orders_during_campaign.columns:
                                            if df_orders_during_campaign[column].dtype == 'object':
                                                if df_orders_during_campaign[column].notnull().any() and isinstance(df_orders_during_campaign[column].iloc[0], object):
                                                    try:
                                                        if hasattr(df_orders_during_campaign[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                                            df_orders_during_campaign[column] = df_orders_during_campaign[column].astype(str)
                                                    except:
                                                        pass
                                        
                                        if not df_orders_during_campaign.empty:
                                            print(f"Found {len(df_orders_during_campaign)} orders during campaign period")
                                            
                                    except Exception as e_campaign_orders:
                                        print(f"Error getting orders during campaign: {e_campaign_orders}")
                                        
                                        # Try to fix the query
                                        fix_campaign_orders_prompt = f"""
                                        The SQL query failed with error: {str(e_campaign_orders)}
                                        
                                        Original query:
                                        {campaign_orders_query}
                                        
                                        This is for PostgreSQL database. 
                                        Please provide a fixed version of this query that will run successfully.
                                        
                                        TABLES AND COLUMNS:
                                        - Orders table '{orders_table}' has columns: {', '.join(orders_columns)}
                                        {f"- Order-product junction table '{order_product_table}' has columns: {', '.join(order_product_columns)}" if order_product_table else ""}
                                        
                                        Important: Return ONLY the SQL code itself without any explanation, markdown formatting or backticks.
                                        For PostgreSQL, make sure to use double quotes for column names with capital letters or special characters.
                                        """
                                        
                                        try:
                                            fix_response = self.llm.invoke(fix_campaign_orders_prompt)
                                            fixed_campaign_orders_query = fix_response.content.strip()
                                            
                                            # Clean up the fixed query
                                            if fixed_campaign_orders_query.startswith("```"):
                                                first_line_end = fixed_campaign_orders_query.find("\n")
                                                if first_line_end != -1:
                                                    fixed_campaign_orders_query = fixed_campaign_orders_query[first_line_end+1:]
                                                if fixed_campaign_orders_query.endswith("```"):
                                                    fixed_campaign_orders_query = fixed_campaign_orders_query[:-3].strip()
                                            
                                            print(f"Attempting fixed campaign period orders query: {fixed_campaign_orders_query}")
                                            
                                            # Add a LIMIT if not present
                                            if "LIMIT" not in fixed_campaign_orders_query.upper():
                                                fixed_campaign_orders_query = fixed_campaign_orders_query.rstrip(';') + " LIMIT 30;"
                                            
                                            df_orders_during_campaign = pd.read_sql(text(fixed_campaign_orders_query), conn)
                                            
                                            # Convert UUID columns to strings
                                            for column in df_orders_during_campaign.columns:
                                                if df_orders_during_campaign[column].dtype == 'object':
                                                    if df_orders_during_campaign[column].notnull().any() and isinstance(df_orders_during_campaign[column].iloc[0], object):
                                                        try:
                                                            if hasattr(df_orders_during_campaign[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                                                df_orders_during_campaign[column] = df_orders_during_campaign[column].astype(str)
                                                        except:
                                                            pass
                                            
                                            campaign_orders_query = fixed_campaign_orders_query
                                            
                                        except Exception as e2:
                                            print(f"Error with fixed campaign period orders query: {e2}")
                            except Exception as e_dates:
                                print(f"Error processing campaign dates: {e_dates}")
                    
                    except Exception as e_campaign_period:
                        print(f"Error determining campaign period: {e_campaign_period}")
                
                execution_time = time.time() - start_time
                print(f"Data retrieved in {execution_time:.2f}s")
                
                # Check if we have enough data to make an analysis
                if df_insights.empty and df_orders.empty:
                    return {
                        "success": False,
                        "error": "Could not retrieve campaign insights or product orders data",
                        "product_data": df_product if not df_product.empty else None
                    }
                
                # Format the queries for display
                full_query = (
                    f"/* Campaign Insights Query */\n{campaign_insights_query}\n\n"
                    f"/* Product Query */\n{product_query}\n\n"
                    f"/* Orders Query */\n{orders_query}"
                )
                
                # Generate analysis based on the retrieved data and incorporate the user question
                analysis_prompt = f"""
                Analiza la relación entre la campaña ID {campaign_id} y el producto ID {product_id} 
                basándote en los datos de la campaña y los pedidos del producto.
                
                {f"PREGUNTA DEL USUARIO: {user_question}" if user_question else ""}

                DATOS DE LA CAMPAÑA:
                {"No se encontraron datos de la campaña" if df_insights.empty else f"Campaña ID: {campaign_id}"}
                {df_insights.head(7).to_string() if not df_insights.empty else ""}
                
                DATOS DEL PRODUCTO:
                {"No se encontraron datos del producto" if df_product.empty else f"Producto ID: {product_id}"}
                {df_product.to_string() if not df_product.empty else ""}
                
                DATOS DE PEDIDOS DEL PRODUCTO:
                {"No se encontraron datos de pedidos" if df_orders.empty else f"Se encontraron {len(df_orders)} pedidos para el producto"}
                {df_orders.head(7).to_string() if not df_orders.empty else ""}
                
                {"PEDIDOS DURANTE EL PERÍODO DE LA CAMPAÑA:" if not df_orders_during_campaign.empty else ""}
                {df_orders_during_campaign.head(7).to_string() if not df_orders_during_campaign.empty else ""}
                
                Basándote en los datos anteriores, proporciona un análisis de marketing que:
                1. Resuma las métricas de rendimiento de la campaña a partir de los datos
                2. Analice los patrones de pedidos del producto, especialmente durante el período de la campaña
                3. Identifique cualquier correlación entre las actividades de la campaña y las ventas del producto
                4. Sugiera si la campaña tuvo un impacto positivo, negativo o neutral en el producto
                5. Recomiende estrategias de marketing futuras para este producto basadas en estos datos
                {f"6. MÁS IMPORTANTE, responde directa y específicamente a la pregunta del usuario: {user_question}" if user_question else ""}
                
                Formatea tu análisis con secciones claras y viñetas para los puntos clave. No proporciones un título grande, solo las secciones.
                """
                
                analysis_response = self.llm.invoke(analysis_prompt)
                analysis = analysis_response.content
                
                return {
                    "success": True,
                    "sql_query": full_query,
                    "user_question": user_question,  # Include the user's question in the return value
                    "campaign_insights": df_insights if not df_insights.empty else None,
                    "product_data": df_product if not df_product.empty else None,
                    "product_orders": df_orders if not df_orders.empty else None,
                    "orders_during_campaign": df_orders_during_campaign if not df_orders_during_campaign.empty else None,
                    "analysis": analysis,
                    "timing": {
                        "data_retrieval": execution_time
                    }
                }
                
        except Exception as e:
            print(f"Error in correlation query execution: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
            
    def analyze_correlation(self, product_id, campaign_id):
        """
        A simple wrapper to execute correlation analysis between a product and campaign
        
        Parameters:
        -----------
        product_id : str
            The ID of the product to analyze
        campaign_id : str
            The ID of the campaign to analyze
            
        Returns:
        --------
        dict
            Results of the correlation analysis
        """
        print(f"Analyzing correlation between product ID {product_id} and campaign ID {campaign_id}")
        
        # This is just a wrapper around execute_correlation_query
        return self.execute_correlation_query(product_id, campaign_id)