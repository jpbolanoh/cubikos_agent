import pandas as pd
import sqlalchemy as sql

def test_connection(db_url):
    """Test the database connection"""
    try:
        engine = sql.create_engine(db_url)
        conn = engine.connect()
        conn.close()
        return True
    except Exception as e:
        return False, str(e)

def get_jsonb_fields(conn, table_name, jsonb_column):
    """
    Extract keys from a JSONB column
    
    Parameters:
    -----------
    conn : sqlalchemy.Connection
        Database connection
    table_name : str
        Name of the table containing the JSONB column
    jsonb_column : str
        Name of the JSONB column
        
    Returns:
    --------
    list
        List of keys found in the JSONB column
    """
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

def execute_sql_query(conn, query):
    """
    Execute a SQL query and return the results as a dataframe
    
    Parameters:
    -----------
    conn : sqlalchemy.Connection
        Database connection
    query : str
        SQL query to execute
        
    Returns:
    --------
    pandas.DataFrame
        Query results
    """
    try:
        from sqlalchemy import text
        return pd.read_sql(text(query), conn)
    except Exception as e:
        raise Exception(f"Error executing SQL query: {e}")

def get_database_schema(conn, table_pattern=None):
    """
    Get the database schema
    
    Parameters:
    -----------
    conn : sqlalchemy.Connection
        Database connection
    table_pattern : str, optional
        Filter tables by pattern
        
    Returns:
    --------
    dict
        Schema information
    """
    try:
        # Get tables
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        tables_df = pd.read_sql_query(tables_query, conn)
        
        # Filter tables if pattern provided
        if table_pattern:
            tables_df = tables_df[tables_df['table_name'].str.contains(table_pattern)]
        
        schema = {}
        
        # Get columns for each table
        for table in tables_df['table_name']:
            columns_query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{table}'
            ORDER BY ordinal_position;
            """
            columns_df = pd.read_sql_query(columns_query, conn)
            
            # Add to schema
            schema[table] = columns_df.to_dict(orient='records')
        
        return schema
    except Exception as e:
        print(f"Error getting database schema: {e}")
        return {}