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
            SELECT * FROM {main_product_table}
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
            SELECT * FROM {main_campaign_table}
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
    
    def execute_correlation_query(self, product_id, campaign_id):
        """
        Execute a specialized analysis of the correlation between a specific product and campaign
        
        Parameters:
        -----------
        product_id : str
            The ID of the product to analyze
        campaign_id : str
            The ID of the campaign to analyze
            
        Returns:
        --------
        dict
            A dictionary containing the results of the analysis
        """
        try:
            print(f"Analyzing correlation between product ID {product_id} and campaign ID {campaign_id}")
            
            # Find relevant product and campaign tables
            product_tables = [t for t in self.all_tables if "product" in t.lower()]
            campaign_tables = [t for t in self.all_tables if "campaign" in t.lower()]
            
            if not product_tables or not campaign_tables:
                return {
                    "success": False,
                    "error": "Could not find required product or campaign tables in the database."
                }
                
            # Find main tables
            product_table = next((t for t in product_tables if t == "inte_dropi_product"), product_tables[0])
            campaign_table = next((t for t in campaign_tables if t == "inte_meta_campaign"), campaign_tables[0])
            campaign_insights_table = next((t for t in campaign_tables if "insight" in t.lower()), None)
            
            # Get the actual column names for these tables
            engine = sql.create_engine(self.db._engine.url)
            with engine.connect() as conn:
                from sqlalchemy import text
                
                # Get product table columns
                product_cols_query = f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{product_table}' ORDER BY ordinal_position;
                """
                product_cols = pd.read_sql(text(product_cols_query), conn)['column_name'].tolist()
                
                # Get campaign table columns
                campaign_cols_query = f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{campaign_table}' ORDER BY ordinal_position;
                """
                campaign_cols = pd.read_sql(text(campaign_cols_query), conn)['column_name'].tolist()
                
                # Get insights table columns if available
                insights_cols = []
                if campaign_insights_table:
                    insights_cols_query = f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = '{campaign_insights_table}' ORDER BY ordinal_position;
                    """
                    insights_cols = pd.read_sql(text(insights_cols_query), conn)['column_name'].tolist()
            
                # Generate a query prompt with precise schema information
                correlation_prompt = f"""
                You are a SQL expert working with PostgreSQL. Create a query to analyze the correlation between 
                product ID '{product_id}' and campaign ID '{campaign_id}'.
                
                TABLES AND COLUMNS:
                - {product_table}: {', '.join(product_cols[:20])}
                - {campaign_table}: {', '.join(campaign_cols[:20])}
                """
                
                if campaign_insights_table and insights_cols:
                    correlation_prompt += f"\n- {campaign_insights_table}: {', '.join(insights_cols[:20])}"
                    
                    # Add JSONB info if available
                    if campaign_insights_table in self.jsonb_info and 'data' in insights_cols:
                        jsonb_keys = self.jsonb_info.get(campaign_insights_table, {}).get('data', [])
                        if jsonb_keys:
                            correlation_prompt += f"\n  JSONB 'data' column keys: {', '.join(jsonb_keys[:10])}"
                            correlation_prompt += "\n  Access with: data->>'key_name'"
                
                correlation_prompt += """
                
                IMPORTANT: 
                1. ONLY use tables and columns in the lists above.
                2. Create a useful query to analyze the relationship between this product and campaign.
                3. Return JUST the SQL query with no explanation.
                4. Use proper PostgreSQL syntax.
                5. Make sure to add a LIMIT clause at the end of your query.
                """
                
                # Generate the SQL query
                start_time = time.time()
                sql_response = self.llm.invoke(correlation_prompt)
                sql_query = sql_response.content.strip()
                query_gen_time = time.time() - start_time
                print(f"Generated correlation SQL in {query_gen_time:.2f}s")
                
                # Clean up the SQL query
                if sql_query.startswith("```"):
                    first_line_end = sql_query.find("\n")
                    if first_line_end != -1:
                        sql_query = sql_query[first_line_end+1:]
                    
                    if sql_query.endswith("```"):
                        sql_query = sql_query[:-3].strip()
                
                # Add a LIMIT if not present
                if "LIMIT" not in sql_query.upper():
                    sql_query = sql_query.rstrip(';') + " LIMIT 100;"
                
                print(f"Executing correlation query: {sql_query}")
                
                # Execute the query
                try:
                    start_time = time.time()
                    df = pd.read_sql(text(sql_query), conn)
                    execution_time = time.time() - start_time
                    print(f"Query executed in {execution_time:.2f}s, returned {len(df)} rows")
                    
                    # Convert UUID columns to strings to avoid PyArrow errors
                    for column in df.columns:
                        if df[column].dtype == 'object':
                            if df[column].notnull().any() and isinstance(df[column].iloc[0], object):
                                try:
                                    if hasattr(df[column].iloc[0], 'hex'):  # Check if it's UUID-like
                                        df[column] = df[column].astype(str)
                                except:
                                    pass
                    
                except Exception as e:
                    print(f"Error executing correlation query: {e}")
                    
                    # Try to fix the query using the LLM
                    fix_prompt = f"""
                    The SQL query failed with error: {str(e)}
                    
                    Original query:
                    {sql_query}
                    
                    This is for PostgreSQL database. 
                    Fix the query to analyze correlation between product ID '{product_id}' and campaign ID '{campaign_id}'.
                    Return ONLY the fixed SQL code without any markdown formatting or backticks.
                    
                    Available tables:
                    - Product table: {product_table} with columns: {', '.join(product_cols[:15])}
                    - Campaign table: {campaign_table} with columns: {', '.join(campaign_cols[:15])}
                    """
                    
                    if campaign_insights_table:
                        fix_prompt += f"- Insights table: {campaign_insights_table} with columns: {', '.join(insights_cols[:15])}"
                    
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
                        df = pd.read_sql(text(fixed_sql), conn)
                        sql_query = fixed_sql  # Update the query that worked
                        
                    except Exception as e2:
                        print(f"Error with fixed query: {e2}")
                        
                        # Fall back to separate basic queries
                        product_id_col = next((c for c in product_cols if c.lower() == 'id'), 'id')
                        campaign_id_col = next((c for c in campaign_cols if c.lower() == 'id'), 'id')
                        
                        product_query = f"""
                        -- Get product info
                        SELECT *, '{product_id}' as product_id, '{campaign_id}' as campaign_id, 'product' as source 
                        FROM {product_table} 
                        WHERE {product_id_col} = '{product_id}'
                        LIMIT 10;
                        """
                        
                        campaign_query = f"""
                        -- Get campaign info
                        SELECT *, '{product_id}' as product_id, '{campaign_id}' as campaign_id, 'campaign' as source 
                        FROM {campaign_table}
                        WHERE {campaign_id_col} = '{campaign_id}'
                        LIMIT 10;
                        """
                        
                        try:
                            # Execute and combine results from both queries
                            df_product = pd.read_sql(text(product_query), conn)
                            df_campaign = pd.read_sql(text(campaign_query), conn)
                            
                            # Convert UUID columns
                            for df_temp in [df_product, df_campaign]:
                                for column in df_temp.columns:
                                    if df_temp[column].dtype == 'object':
                                        if df_temp[column].notnull().any() and isinstance(df_temp[column].iloc[0], object):
                                            try:
                                                if hasattr(df_temp[column].iloc[0], 'hex'):
                                                    df_temp[column] = df_temp[column].astype(str)
                                            except:
                                                pass
                            
                            # Combine the dataframes
                            df = pd.concat([df_product, df_campaign], axis=0)
                            sql_query = f"{product_query}\n\n{campaign_query}"
                            
                        except Exception as e3:
                            return {
                                "success": False,
                                "error": f"Failed to execute all correlation queries: {str(e3)}",
                                "original_query": sql_query
                            }
                
                # Generate analysis from the data
                if not df.empty:
                    analysis_prompt = f"""
                    Analyze the correlation between product ID {product_id} and campaign ID {campaign_id} 
                    based on the following data:
                    
                    Data shape: {df.shape[0]} rows, {df.shape[1]} columns
                    Columns: {', '.join(df.columns.tolist())}
                    Sample data:
                    {df.head(5).to_string()}
                    
                    Provide an analysis of what these data show about the possible 
                    relationship between this product and campaign. Consider:
                    
                    1. Is there evidence of a direct relationship?
                    2. What metrics suggest performance correlation?
                    3. What business insights can be drawn?
                    4. Any limitations in this analysis?
                    
                    Keep your analysis concise but informative.
                    """
                    
                    analysis_response = self.llm.invoke(analysis_prompt)
                    analysis = analysis_response.content
                else:
                    analysis = "The correlation analysis did not return any data."
                
                return {
                    "success": True,
                    "sql_query": sql_query,
                    "data": df,
                    "analysis": analysis,
                    "timing": {
                        "query_generation": locals().get('query_gen_time', 0),
                        "query_execution": locals().get('execution_time', 0)
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