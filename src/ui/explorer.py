import streamlit as st
import pandas as pd
from sqlalchemy import text
from src.utils.db_utils import get_jsonb_fields

def show_explorer_tab(conn, db_agent):
    """
    Display the Database Explorer tab content
    
    Parameters:
    -----------
    conn : sqlalchemy.Connection
        Database connection
    db_agent : MarketingDBAgent
        The database agent
    """
    st.header("Database Explorer")
    
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
        
        if category_filter != "All":
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
                    
                    # Show sample JSONB data
                    if st.button("Show sample JSONB data"):
                        sample_jsonb_query = f"""
                        SELECT {selected_jsonb}
                        FROM {selected_table}
                        WHERE {selected_jsonb} IS NOT NULL
                        LIMIT 1;
                        """
                        try:
                            sample_jsonb_df = pd.read_sql_query(text(sample_jsonb_query), conn)
                            if not sample_jsonb_df.empty:
                                st.json(sample_jsonb_df.iloc[0, 0])
                        except Exception as e:
                            st.error(f"Error fetching JSONB sample: {e}")
            
            # Show sample data
            if st.button("Show sample data"):
                sample_query = f"SELECT * FROM {selected_table} LIMIT 5"
                sample_df = pd.read_sql_query(text(sample_query), conn)
                st.dataframe(sample_df)
                
                # Add statistics
                if st.checkbox("Show statistics"):
                    st.write("Table statistics:")
                    stats_query = f"""
                    SELECT COUNT(*) as total_rows
                    FROM {selected_table};
                    """
                    stats_df = pd.read_sql_query(text(stats_query), conn)
                    st.write(f"Total rows: {stats_df.iloc[0, 0]}")
                    
                    # For numeric columns, show min, max, avg
                    numeric_columns = columns_df[columns_df['data_type'].isin(['integer', 'numeric', 'double precision', 'real', 'bigint'])]['column_name'].tolist()
                    if numeric_columns:
                        st.write("Numeric column statistics:")
                        stats_list = []
                        for col in numeric_columns[:5]:  # Limit to 5 columns to avoid long queries
                            try:
                                col_stats_query = f"""
                                SELECT 
                                    MIN("{col}") as min,
                                    MAX("{col}") as max,
                                    AVG("{col}") as avg
                                FROM {selected_table}
                                WHERE "{col}" IS NOT NULL;
                                """
                                col_stats_df = pd.read_sql_query(text(col_stats_query), conn)
                                stats_list.append({
                                    "Column": col,
                                    "Min": col_stats_df.iloc[0, 0],
                                    "Max": col_stats_df.iloc[0, 1],
                                    "Avg": col_stats_df.iloc[0, 2]
                                })
                            except Exception as e:
                                print(f"Error getting stats for column {col}: {e}")
                        
                        if stats_list:
                            stats_df = pd.DataFrame(stats_list)
                            st.dataframe(stats_df)
            
            # Show table relationships
            if st.button("Show relationships"):
                fk_query = f"""
                SELECT
                    tc.constraint_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM
                    information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE
                    tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_name = '{selected_table}';
                """
                try:
                    fk_df = pd.read_sql_query(text(fk_query), conn)
                    if not fk_df.empty:
                        st.write("Foreign keys:")
                        st.dataframe(fk_df)
                    else:
                        st.write("No foreign keys found.")
                        
                    # Also show tables referencing this table
                    ref_query = f"""
                    SELECT
                        tc.table_name AS referencing_table,
                        kcu.column_name AS referencing_column,
                        ccu.column_name AS referenced_column
                    FROM
                        information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE
                        tc.constraint_type = 'FOREIGN KEY'
                        AND ccu.table_name = '{selected_table}';
                    """
                    ref_df = pd.read_sql_query(text(ref_query), conn)
                    if not ref_df.empty:
                        st.write("Referenced by:")
                        st.dataframe(ref_df)
                    else:
                        st.write("Not referenced by any table.")
                except Exception as e:
                    st.error(f"Error fetching relationships: {e}")
    except Exception as e:
        st.error(f"Error exploring database: {e}")