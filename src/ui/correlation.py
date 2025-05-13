import streamlit as st
import pandas as pd
import time
from sqlalchemy import text

def show_correlation_tab(db_agent, conn):
    """
    Muestra el contenido de la pestaña de Análisis de Correlación Producto-Campaña
    
    Parámetros:
    -----------
    db_agent : MarketingDBAgent
        El agente de base de datos
    conn : sqlalchemy.Connection
        Conexión a la base de datos
    """
    st.header("Análisis de Correlación entre Productos y Campañas")
    
    st.markdown("""
    Esta herramienta te permite seleccionar un producto y una campaña específicos, y luego analizar su correlación.
    Puedes hacer preguntas sobre cómo se relacionan entre sí, y la IA generará información valiosa.
    """)
    
    # Crear dos columnas para la selección de producto y campaña
    col1, col2 = st.columns(2)
    
    # Selección de producto en la primera columna
    with col1:
        st.subheader("Seleccionar Producto")
        
        # Obtener datos de productos
        with st.spinner("Cargando productos..."):
            products_df = db_agent.get_products()
        
        if products_df.empty:
            st.warning("No se encontraron productos en la base de datos")
            product_id = None
        else:
            # Determinar columnas de ID y nombre del producto
            product_id_col = next((col for col in products_df.columns if 'id' in col.lower()), products_df.columns[0])
            potential_name_cols = [col for col in products_df.columns if any(name in col.lower() for name in ['name', 'title', 'product'])]
            product_name_col = next(iter(potential_name_cols), product_id_col)
            
            # Crear un dataframe de visualización con columnas clave
            display_cols = [product_id_col, product_name_col]
            # Añadir algunas columnas informativas más si existen
            for col in ['price', 'sku', 'category', 'type']:
                matching_cols = [c for c in products_df.columns if col in c.lower()]
                if matching_cols:
                    display_cols.append(matching_cols[0])
            
            # Limitar a productos únicos
            display_products = products_df[display_cols].drop_duplicates().head(100)
            
            st.dataframe(display_products)
            
            # Permitir selección por ID
            product_id = st.text_input("Ingresar ID del Producto", key="product_id_input")
            
            # Mostrar detalles del producto seleccionado si se ingresa ID
            if product_id:
                filtered_product = products_df[products_df[product_id_col] == product_id]
                if not filtered_product.empty:
                    st.success(f"Producto Seleccionado: {filtered_product[product_name_col].values[0]}")
                    with st.expander("Detalles del Producto"):
                        st.json(filtered_product.iloc[0].to_dict())
                else:
                    st.warning(f"No se encontró ningún producto con ID {product_id}")
    
    # Selección de campaña en la segunda columna
    with col2:
        st.subheader("Seleccionar Campaña")
        
        # Obtener datos de campañas
        with st.spinner("Cargando campañas..."):
            campaigns_df = db_agent.get_campaigns()
        
        if campaigns_df.empty:
            st.warning("No se encontraron campañas en la base de datos")
            campaign_id = None
        else:
            # Determinar columnas de ID y nombre de la campaña
            campaign_id_col = next((col for col in campaigns_df.columns if 'id' in col.lower()), campaigns_df.columns[0])
            potential_name_cols = [col for col in campaigns_df.columns if any(name in col.lower() for name in ['name', 'title', 'campaign'])]
            campaign_name_col = next(iter(potential_name_cols), campaign_id_col)
            
            # Crear un dataframe de visualización con columnas clave
            display_cols = [campaign_id_col, campaign_name_col]
            # Añadir algunas columnas informativas más si existen
            for col in ['status', 'budget', 'start_time', 'end_time']:
                matching_cols = [c for c in campaigns_df.columns if col in c.lower()]
                if matching_cols:
                    display_cols.append(matching_cols[0])
            
            # Limitar a campañas únicas
            display_campaigns = campaigns_df[display_cols].drop_duplicates().head(100)
            
            st.dataframe(display_campaigns)
            
            # Permitir selección por ID
            campaign_id = st.text_input("Ingresar ID de la Campaña", key="campaign_id_input")
            
            # Mostrar detalles de la campaña seleccionada si se ingresa ID
            if campaign_id:
                filtered_campaign = campaigns_df[campaigns_df[campaign_id_col] == campaign_id]
                if not filtered_campaign.empty:
                    st.success(f"Campaña Seleccionada: {filtered_campaign[campaign_name_col].values[0]}")
                    with st.expander("Detalles de la Campaña"):
                        st.json(filtered_campaign.iloc[0].to_dict())
                else:
                    st.warning(f"No se encontró ninguna campaña con ID {campaign_id}")
    
    # Sección de Análisis de Correlación
    st.subheader("Análisis")
    
    # Solo mostrar análisis de correlación si se han seleccionado tanto producto como campaña
    if product_id and campaign_id:
        # Pregunta de texto libre sobre correlación
        correlation_question = st.text_area(
            "Haz una pregunta sobre la correlación entre el producto y la campaña seleccionados",
            height=100,
            placeholder="Ejemplo: ¿Cómo ha afectado esta campaña a las ventas de este producto?",
            key="correlation_question_input"
        )
        
        # Si no hay pregunta, sugerir una pregunta predeterminada
        if not correlation_question:
            correlation_question = "¿Cómo se relacionan este producto y esta campaña? Proporciona un análisis completo."
        
        # Botón para enviar pregunta
        if st.button("Analizar Relación", key="submit_correlation_question"):
            with st.spinner("Analizando la relación entre producto y campaña..."):
                # Comprobar si la función execute_correlation_query actualizada está disponible
                if hasattr(db_agent, 'execute_correlation_query'):
                    correlation_result = db_agent.execute_correlation_query(
                        product_id, 
                        campaign_id,
                        user_question=correlation_question
                    )
                    
                    if correlation_result["success"]:
                        # Mostrar la pregunta del usuario de forma destacada
                        st.info(f"**Pregunta:** {correlation_question}")
                        
                        # Ocultar SQL por defecto
                        with st.expander("Ver Consulta SQL", expanded=False):
                            st.code(correlation_result["sql_query"], language="sql")
                        
                        # Sección principal de análisis
                        st.subheader("Análisis de Correlación")
                        st.markdown(correlation_result["analysis"])
                        
                        # Sección de datos en expandibles
                        with st.expander("Ver datos de la campaña", expanded=False):
                            if "campaign_insights" in correlation_result and isinstance(correlation_result["campaign_insights"], pd.DataFrame) and not correlation_result["campaign_insights"].empty:
                                st.subheader("Información de la Campaña")
                                st.dataframe(correlation_result["campaign_insights"])
                                
                                # Añadir botón de descarga
                                csv = correlation_result["campaign_insights"].to_csv(index=False)
                                st.download_button(
                                    label="Descargar información de campaña como CSV",
                                    data=csv,
                                    file_name="informacion_campana.csv",
                                    mime="text/csv",
                                    key=f"download_campaign_insights_{product_id}_{campaign_id}"
                                )
                        
                        with st.expander("Ver datos del producto", expanded=False):
                            if "product_data" in correlation_result and isinstance(correlation_result["product_data"], pd.DataFrame) and not correlation_result["product_data"].empty:
                                st.subheader("Información del Producto")
                                st.dataframe(correlation_result["product_data"])
                        
                        with st.expander("Ver pedidos del producto", expanded=False):
                            if "product_orders" in correlation_result and isinstance(correlation_result["product_orders"], pd.DataFrame) and not correlation_result["product_orders"].empty:
                                st.subheader("Pedidos del Producto")
                                st.dataframe(correlation_result["product_orders"])
                                
                                # Añadir botón de descarga
                                csv = correlation_result["product_orders"].to_csv(index=False)
                                st.download_button(
                                    label="Descargar pedidos del producto como CSV",
                                    data=csv,
                                    file_name="pedidos_producto.csv",
                                    mime="text/csv",
                                    key=f"download_product_orders_{product_id}_{campaign_id}"
                                )
                        
                        with st.expander("Ver pedidos durante la campaña", expanded=False):
                            if "orders_during_campaign" in correlation_result and isinstance(correlation_result["orders_during_campaign"], pd.DataFrame) and not correlation_result["orders_during_campaign"].empty:
                                st.subheader("Pedidos Durante el Período de la Campaña")
                                st.dataframe(correlation_result["orders_during_campaign"])
                                
                                # Añadir botón de descarga
                                csv = correlation_result["orders_during_campaign"].to_csv(index=False)
                                st.download_button(
                                    label="Descargar pedidos durante campaña como CSV",
                                    data=csv,
                                    file_name="pedidos_durante_campana.csv",
                                    mime="text/csv",
                                    key=f"download_campaign_orders_{product_id}_{campaign_id}"
                                )
                    else:
                        st.error("La consulta falló")
                        if "error" in correlation_result:
                            st.error(correlation_result["error"])
                        else:
                            st.error("Ocurrió un error desconocido")
                else:
                    # Usar la implementación anterior si execute_correlation_query no está disponible
                    # Generar una consulta especializada para la pregunta específica sobre este producto y campaña
                    custom_prompt = f"""
                    Crea una consulta SQL para responder a esta pregunta específica sobre la correlación entre 
                    el producto ID {product_id} y la campaña ID {campaign_id}:
                    
                    "{correlation_question}"
                    
                    Tablas de productos disponibles: {', '.join([t for t in db_agent.all_tables if 'product' in t.lower()])}
                    Tablas de campañas disponibles: {', '.join([t for t in db_agent.all_tables if 'campaign' in t.lower()])}
                    
                    Devuelve SOLO la consulta SQL sin explicaciones.
                    """
                    
                    # Generar SQL para esta pregunta específica
                    sql_response = db_agent.llm.invoke(custom_prompt)
                    custom_sql = sql_response.content.strip()
                    
                    # Limpiar el SQL
                    if custom_sql.startswith("```"):
                        first_line_end = custom_sql.find("\n")
                        if first_line_end != -1:
                            custom_sql = custom_sql[first_line_end+1:]
                        
                        if custom_sql.endswith("```"):
                            custom_sql = custom_sql[:-3].strip()
                    
                    # Ejecutar la consulta usando nuestro ejecutor de consulta de correlación personalizado
                    try:
                        # Primero intenta ejecutar directamente con la conexión SQL
                        try:
                            with conn.begin():
                                # Asegúrate de que el SQL tenga un LIMIT
                                if "LIMIT" not in custom_sql.upper():
                                    custom_sql = custom_sql.rstrip(';') + " LIMIT 100;"
                                
                                custom_df = pd.read_sql(text(custom_sql), conn)
                                
                                # Generar análisis
                                analysis_prompt = f"""
                                Analiza estos resultados para la correlación entre el producto ID {product_id} y la campaña ID {campaign_id}:
                                
                                Pregunta: {correlation_question}
                                
                                Datos: {custom_df.head(10).to_string()}
                                
                                Proporciona información sobre lo que estos datos muestran con respecto a la relación entre el producto y la campaña,
                                y responde directamente a la pregunta anterior.
                                """
                                
                                analysis_response = db_agent.llm.invoke(analysis_prompt)
                                
                                correlation_result = {
                                    "success": True,
                                    "sql_query": custom_sql,
                                    "custom_data": custom_df,  # Usar un nombre de clave diferente
                                    "analysis": analysis_response.content
                                }
                        except Exception as sql_err:
                            st.warning(f"La ejecución SQL directa falló: {sql_err}")
                            st.info("Probando método de ejecución alternativo...")
                            
                            # Si la ejecución directa falla, usa el método analyze_correlation
                            correlation_result = db_agent.analyze_correlation(product_id, campaign_id)
                    
                        if correlation_result and correlation_result["success"]:
                            # Mostrar la pregunta del usuario de forma destacada
                            st.info(f"**Pregunta:** {correlation_question}")
                            
                            # Ocultar SQL por defecto
                            with st.expander("Ver Consulta SQL", expanded=False):
                                st.code(correlation_result["sql_query"], language="sql")
                            
                            # Sección principal de análisis
                            st.subheader("Análisis de Correlación")
                            st.markdown(correlation_result["analysis"])
                            
                            # Sección de datos en expandibles
                            with st.expander("Ver resultados de la consulta", expanded=False):
                                if "custom_data" in correlation_result and isinstance(correlation_result["custom_data"], pd.DataFrame) and not correlation_result["custom_data"].empty:
                                    st.subheader("Resultados de la Consulta")
                                    st.dataframe(correlation_result["custom_data"])
                                    
                                    # Añadir botón de descarga
                                    csv = correlation_result["custom_data"].to_csv(index=False)
                                    st.download_button(
                                        label="Descargar resultados como CSV",
                                        data=csv,
                                        file_name="resultados_consulta_personalizada.csv",
                                        mime="text/csv",
                                        key=f"download_custom_results_{product_id}_{campaign_id}"
                                    )
                            
                            with st.expander("Ver datos de la campaña", expanded=False):
                                if "campaign_insights" in correlation_result and isinstance(correlation_result["campaign_insights"], pd.DataFrame) and not correlation_result["campaign_insights"].empty:
                                    st.subheader("Información de la Campaña")
                                    st.dataframe(correlation_result["campaign_insights"])
                                    
                                    # Añadir botón de descarga
                                    csv = correlation_result["campaign_insights"].to_csv(index=False)
                                    st.download_button(
                                        label="Descargar información de campaña como CSV",
                                        data=csv,
                                        file_name="informacion_campana.csv",
                                        mime="text/csv",
                                        key=f"download_campaign_insights_alt_{product_id}_{campaign_id}"
                                    )
                            
                            with st.expander("Ver datos del producto", expanded=False):
                                if "product_data" in correlation_result and isinstance(correlation_result["product_data"], pd.DataFrame) and not correlation_result["product_data"].empty:
                                    st.subheader("Información del Producto")
                                    st.dataframe(correlation_result["product_data"])
                            
                            with st.expander("Ver pedidos del producto", expanded=False):
                                if "product_orders" in correlation_result and isinstance(correlation_result["product_orders"], pd.DataFrame) and not correlation_result["product_orders"].empty:
                                    st.subheader("Pedidos del Producto")
                                    st.dataframe(correlation_result["product_orders"])
                                    
                                    # Añadir botón de descarga
                                    csv = correlation_result["product_orders"].to_csv(index=False)
                                    st.download_button(
                                        label="Descargar pedidos del producto como CSV",
                                        data=csv,
                                        file_name="pedidos_producto.csv",
                                        mime="text/csv",
                                        key=f"download_product_orders_alt_{product_id}_{campaign_id}"
                                    )
                        else:
                            st.error("La consulta falló")
                            if correlation_result and "error" in correlation_result:
                                st.error(correlation_result["error"])
                            else:
                                st.error("Ocurrió un error desconocido")
                    except Exception as e:
                        st.error(f"Error al ejecutar la consulta personalizada: {e}")
    else:
        st.info("Por favor, selecciona tanto un producto como una campaña para analizar su correlación.")