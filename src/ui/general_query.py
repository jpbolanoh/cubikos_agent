import streamlit as st
import pandas as pd
import time

def show_general_query_tab(db_agent, debug_mode=False):
    """
    Display the Natural Language Query tab content
    
    Parameters:
    -----------
    db_agent : MarketingDBAgent
        The database agent
    debug_mode : bool
        Whether to show debug information
    """
    st.header("Natural Language Query")
    
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

    if st.button("Submit Query", key="submit_general_query"):
        if not question:
            st.warning("Please enter a question")
        else:
            with st.spinner("Processing your question..."):
                start_time = time.time()
                
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