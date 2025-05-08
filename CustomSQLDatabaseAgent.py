from ai_data_science_team.agents import SQLDatabaseAgent

class CustomSQLDatabaseAgent(SQLDatabaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def ainvoke_agent(self, user_instructions=None, max_retries=3, retry_count=0, **kwargs):
        await super().ainvoke_agent(user_instructions, max_retries, retry_count, **kwargs)
        
        # After the original agent runs, replace the SQL function with our custom one
        if self.response and "sql_query_code" in self.response:
            sql_query_code = self.response["sql_query_code"]
            custom_function = f"""
def sql_database_pipeline(connection):
    import pandas as pd
    import sqlalchemy as sql
    
    # Create a connection if needed
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection
    
    try:
        # Try to rollback any failed transaction first
        if hasattr(conn, 'execute'):
            try:
                conn.execute(sql.text("ROLLBACK"))
            except:
                # If direct rollback fails, try using a transaction object
                try:
                    transaction = conn.begin()
                    transaction.rollback()
                except:
                    pass
        
        # Create a fresh transaction for our query
        transaction = conn.begin()
        
        sql_query = '''
        {sql_query_code}
        '''
        
        # Execute the query using the transaction
        result = pd.read_sql(sql_query, conn)
        transaction.commit()
        return result
    
    except Exception as e:
        # If an error occurs, ensure we rollback
        try:
            transaction.rollback()
        except:
            pass
        
        # Try a different approach if the first one failed
        try:
            # Get a completely fresh connection to avoid transaction issues
            if is_engine:
                fresh_conn = connection.connect()
            else:
                engine = connection.engine
                fresh_conn = engine.connect()
                
            sql_query = '''
            {sql_query_code}
            '''
            
            result = pd.read_sql(sql_query, fresh_conn)
            fresh_conn.close()
            return result
        except Exception as e2:
            # If all approaches fail, raise the original error
            raise e
"""
            self.response["sql_database_function"] = custom_function
            
            # Now we need to execute the updated function
            from ai_data_science_team.templates import node_func_execute_agent_from_sql_connection
            
            # Execute with the new function
            is_engine = isinstance(self._params["connection"], sql.engine.base.Engine)
            conn = self._params["connection"].connect() if is_engine else self._params["connection"]
            
            # Create a state-like object to pass to the execution function
            state = {
                "sql_database_function": custom_function,
                "sql_database_function_name": "sql_database_pipeline"
            }
            
            result = node_func_execute_agent_from_sql_connection(
                state=state,
                connection=conn,
                result_key="data_sql",
                error_key="sql_database_error",
                code_snippet_key="sql_database_function",
                agent_function_name="sql_database_pipeline",
                post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
                error_message_prefix="An error occurred during executing the sql database pipeline: "
            )
            
            # Update the response with the new result
            if "data_sql" in result:
                self.response["data_sql"] = result["data_sql"]