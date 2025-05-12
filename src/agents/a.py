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