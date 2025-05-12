import streamlit as st
import pandas as pd
import time
from sqlalchemy import text

def show_direct_sql_tab(conn):
    """
    Display the Direct SQL Query tab content
    
    Parameters:
    -----------
    conn : sqlalchemy.Connection
        Database connection
    """
    st.header("Direct SQL Query Tool")
    
    # SQL history
    if "sql_history" not in st.session_state:
        st.session_state.sql_history = []
    
    # Show history in expander
    with st.expander("SQL History", expanded=False):
        if st.session_state.sql_history:
            for i, query in enumerate(st.session_state.sql_history):
                if st.button(f"Use query {i+1}", key=f"use_history_{i}"):
                    st.session_state.direct_sql = query
        else:
            st.write("No queries in history yet.")
    
    # Initialize session state for SQL input
    if "direct_sql" not in st.session_state:
        st.session_state.direct_sql = ""
    
    # SQL input with value from session state
    direct_sql = st.text_area("Enter SQL Query", height=150, value=st.session_state.direct_sql, key="sql_input")
    
    # Query templates
    with st.expander("Query Templates", expanded=False):
        template_options = {
            "Select all columns": "SELECT * FROM table_name LIMIT 10;",
            "Count rows": "SELECT COUNT(*) FROM table_name;",
            "Filter by condition": "SELECT * FROM table_name WHERE column_name = 'value' LIMIT 10;",
            "Join tables": """
SELECT t1.column1, t1.column2, t2.column1
FROM table1 t1
JOIN table2 t2 ON t1.id = t2.table1_id
LIMIT 10;
""",
            "Group by and aggregate": """
SELECT column1, COUNT(*), AVG(numeric_column)
FROM table_name
GROUP BY column1
ORDER BY COUNT(*) DESC
LIMIT 10;
"""
        }
        
        for template_name, template_sql in template_options.items():
            if st.button(template_name, key=f"template_{template_name}"):
                st.session_state.direct_sql = template_sql
                st.experimental_rerun()
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        execute_btn = st.button("Execute SQL")
        
    with col2:
        explain_query = st.checkbox("Explain query plan")
    
    if execute_btn:
        if not direct_sql:
            st.warning("Please enter a SQL query")
        else:
            try:
                with st.spinner("Executing query..."):
                    start_time = time.time()
                    
                    # Add to history if not already there
                    if direct_sql not in st.session_state.sql_history:
                        st.session_state.sql_history.append(direct_sql)
                        # Keep only the latest 5 queries
                        if len(st.session_state.sql_history) > 5:
                            st.session_state.sql_history.pop(0)
                    
                    if explain_query:
                        # Execute EXPLAIN
                        explain_sql = f"EXPLAIN ANALYZE {direct_sql}"
                        explain_result = conn.execute(text(explain_sql))
                        explain_rows = explain_result.fetchall()
                        
                        st.subheader("Query Plan")
                        plan_text = "\n".join([row[0] for row in explain_rows])
                        st.code(plan_text)
                    
                    # Execute the actual query
                    result_df = pd.read_sql_query(text(direct_sql), conn)
                    
                st.success(f"Query executed in {time.time() - start_time:.2f} seconds")
                st.write(f"Returned {len(result_df)} rows")
                
                # Display results
                st.subheader("Results")
                st.dataframe(result_df)
                
                # Result actions
                col1, col2 = st.columns(2)
                
                with col1:
                    # Add download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name="direct_query_results.csv",
                        mime="text/csv",
                    )
                
                with col2:
                    # Show summary statistics
                    if st.button("Show summary statistics"):
                        st.subheader("Summary Statistics")
                        
                        # Check if there are numeric columns
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        
                        if numeric_cols:
                            st.write("Numeric column statistics:")
                            st.dataframe(result_df[numeric_cols].describe())
                        
                        # Column data types
                        st.write("Column Data Types:")
                        dtypes_df = pd.DataFrame({
                            'Column': result_df.columns,
                            'Data Type': result_df.dtypes.astype(str),
                            'Non-Null Count': result_df.count().values,
                            'Null Count': result_df.isna().sum().values
                        })
                        st.dataframe(dtypes_df)
                        
                        # Unique values in categorical columns
                        categorical_cols = result_df.select_dtypes(include=['object']).columns.tolist()
                        if categorical_cols:
                            st.write("Unique values in categorical columns:")
                            for col in categorical_cols[:5]:  # Limit to 5 columns
                                if result_df[col].nunique() < 10:  # Only show if less than 10 unique values
                                    st.write(f"{col}: {result_df[col].unique()}")
            except Exception as e:
                st.error(f"Error executing SQL: {e}")