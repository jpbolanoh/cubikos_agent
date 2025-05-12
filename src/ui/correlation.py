import streamlit as st
import pandas as pd
import time
from sqlalchemy import text

def show_correlation_tab(db_agent, conn):
    """
    Display the Product-Campaign Correlation tab content
    
    Parameters:
    -----------
    db_agent : MarketingDBAgent
        The database agent
    conn : sqlalchemy.Connection
        Database connection
    """
    st.header("Product-Campaign Correlation Analysis")
    
    st.markdown("""
    This tool allows you to select a specific product and campaign, then analyze their correlation.
    You can ask questions about how they relate to each other, and the AI will generate insights.
    """)
    
    # Create two columns for product and campaign selection
    col1, col2 = st.columns(2)
    
    # Product selection in the first column
    with col1:
        st.subheader("Select Product")
        
        # Get product data
        with st.spinner("Loading products..."):
            products_df = db_agent.get_products()
        
        if products_df.empty:
            st.warning("No products found in the database")
            product_id = None
        else:
            # Determine product ID and name columns
            product_id_col = next((col for col in products_df.columns if 'id' in col.lower()), products_df.columns[0])
            potential_name_cols = [col for col in products_df.columns if any(name in col.lower() for name in ['name', 'title', 'product'])]
            product_name_col = next(iter(potential_name_cols), product_id_col)
            
            # Create a display dataframe with key columns
            display_cols = [product_id_col, product_name_col]
            # Add a few more informative columns if they exist
            for col in ['price', 'sku', 'category', 'type']:
                matching_cols = [c for c in products_df.columns if col in c.lower()]
                if matching_cols:
                    display_cols.append(matching_cols[0])
            
            # Limit to unique products
            display_products = products_df[display_cols].drop_duplicates().head(100)
            
            st.dataframe(display_products)
            
            # Allow selection by ID
            product_id = st.text_input("Enter Product ID", key="product_id_input")
            
            # Show details of selected product if ID is entered
            if product_id:
                filtered_product = products_df[products_df[product_id_col] == product_id]
                if not filtered_product.empty:
                    st.success(f"Selected Product: {filtered_product[product_name_col].values[0]}")
                    with st.expander("Product Details"):
                        st.json(filtered_product.iloc[0].to_dict())
                else:
                    st.warning(f"No product found with ID {product_id}")
    
    # Campaign selection in the second column
    with col2:
        st.subheader("Select Campaign")
        
        # Get campaign data
        with st.spinner("Loading campaigns..."):
            campaigns_df = db_agent.get_campaigns()
        
        if campaigns_df.empty:
            st.warning("No campaigns found in the database")
            campaign_id = None
        else:
            # Determine campaign ID and name columns
            campaign_id_col = next((col for col in campaigns_df.columns if 'id' in col.lower()), campaigns_df.columns[0])
            potential_name_cols = [col for col in campaigns_df.columns if any(name in col.lower() for name in ['name', 'title', 'campaign'])]
            campaign_name_col = next(iter(potential_name_cols), campaign_id_col)
            
            # Create a display dataframe with key columns
            display_cols = [campaign_id_col, campaign_name_col]
            # Add a few more informative columns if they exist
            for col in ['status', 'budget', 'start_time', 'end_time']:
                matching_cols = [c for c in campaigns_df.columns if col in c.lower()]
                if matching_cols:
                    display_cols.append(matching_cols[0])
            
            # Limit to unique campaigns
            display_campaigns = campaigns_df[display_cols].drop_duplicates().head(100)
            
            st.dataframe(display_campaigns)
            
            # Allow selection by ID
            campaign_id = st.text_input("Enter Campaign ID", key="campaign_id_input")
            
            # Show details of selected campaign if ID is entered
            if campaign_id:
                filtered_campaign = campaigns_df[campaigns_df[campaign_id_col] == campaign_id]
                if not filtered_campaign.empty:
                    st.success(f"Selected Campaign: {filtered_campaign[campaign_name_col].values[0]}")
                    with st.expander("Campaign Details"):
                        st.json(filtered_campaign.iloc[0].to_dict())
                else:
                    st.warning(f"No campaign found with ID {campaign_id}")
    
    # Correlation Analysis Section
    st.subheader("Analysis")
    
    # Only show correlation analysis if both product and campaign are selected
    if product_id and campaign_id:
        # Free text question about correlation
        correlation_question = st.text_area(
            "Ask a question about the correlation between the selected product and campaign",
            height=100,
            placeholder="Example: How has this campaign affected the sales of this product?"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Button for quick correlation analysis
            if st.button("Run Correlation Analysis", key="run_correlation"):
                with st.spinner("Analyzing correlation..."):
                    correlation_result = db_agent.analyze_correlation(product_id, campaign_id)
                    
                    if correlation_result["success"]:
                        with st.expander("SQL Query", expanded=True):
                            st.code(correlation_result["sql_query"], language="sql")
                        
                        if isinstance(correlation_result["data"], pd.DataFrame) and not correlation_result["data"].empty:
                            st.subheader("Correlation Data")
                            st.dataframe(correlation_result["data"])
                            
                            # Add download button
                            csv = correlation_result["data"].to_csv(index=False)
                            st.download_button(
                                label="Download correlation data as CSV",
                                data=csv,
                                file_name="correlation_results.csv",
                                mime="text/csv",
                            )
                        
                        st.subheader("Correlation Analysis")
                        st.markdown(correlation_result["analysis"])
                    else:
                        st.error("Correlation analysis failed")
                        st.error(correlation_result["error"])
        
        with col2:
            # Button for custom correlation question
            if correlation_question and st.button("Submit Question", key="submit_correlation_question"):
                with st.spinner("Processing your correlation question..."):
                    # Generate a specialized query for the specific question about this product and campaign
                    custom_prompt = f"""
                    Create a SQL query to answer this specific question about the correlation between 
                    product ID {product_id} and campaign ID {campaign_id}:
                    
                    "{correlation_question}"
                    
                    Available product tables: {', '.join([t for t in db_agent.all_tables if 'product' in t.lower()])}
                    Available campaign tables: {', '.join([t for t in db_agent.all_tables if 'campaign' in t.lower()])}
                    
                    Return ONLY the SQL query without explanations.
                    """
                    
                    # Generate SQL for this specific question
                    sql_response = db_agent.llm.invoke(custom_prompt)
                    custom_sql = sql_response.content.strip()
                    
                    # Clean up the SQL
                    if custom_sql.startswith("```"):
                        first_line_end = custom_sql.find("\n")
                        if first_line_end != -1:
                            custom_sql = custom_sql[first_line_end+1:]
                        
                        if custom_sql.endswith("```"):
                            custom_sql = custom_sql[:-3].strip()
                    
                    # Execute the query using our correlation query executor
                    correlation_result = db_agent.execute_correlation_query(custom_sql, product_id, campaign_id)
                    
                    if correlation_result["success"]:
                        with st.expander("SQL Query", expanded=True):
                            st.code(correlation_result["sql_query"], language="sql")
                        
                        if isinstance(correlation_result["data"], pd.DataFrame) and not correlation_result["data"].empty:
                            st.subheader("Results")
                            st.dataframe(correlation_result["data"])
                            
                            # Add download button
                            csv = correlation_result["data"].to_csv(index=False)
                            st.download_button(
                                label="Download data as CSV",
                                data=csv,
                                file_name="correlation_question_results.csv",
                                mime="text/csv",
                            )
                        
                        st.subheader("Analysis")
                        st.markdown(correlation_result["analysis"])
                    else:
                        st.error("Query failed")
                        st.error(correlation_result["error"])
    else:
        st.info("Please select both a product and a campaign to analyze their correlation.")