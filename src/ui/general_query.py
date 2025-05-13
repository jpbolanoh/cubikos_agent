import streamlit as st
import pandas as pd
import time

def show_general_query_tab(db_agent, debug_mode=False):
    """
    Muestra la interfaz de chat para consultas en lenguaje natural
    
    Parámetros:
    -----------
    db_agent : MarketingDBAgent
        El agente de base de datos
    debug_mode : bool
        Si se debe mostrar información de depuración
    """
    # Configuración de página completa para maximizar espacio
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1dp5vir {
        width: 100%;
    }
    .stChatInput {
        padding-bottom: 5px;
    }
    .stMarkdown p {
        margin-bottom: 0px;
    }
    .stDataFrame {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("Copernico")
    
    # Inicializar el historial de mensajes en el estado de la sesión si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente de análisis de datos. Puedes hacerme cualquier pregunta sobre tus datos de marketing y te ayudaré a encontrar la información que necesitas."}
        ]
    
    # Mostrar ejemplos de preguntas en una barra lateral
    with st.sidebar:
        st.subheader("Ejemplos de Preguntas")
        st.markdown("""
        - Muéstrame 5 filas de inte_dropi_order
        - ¿Cuáles son las 10 campañas con mayor gasto este mes?
        - Análisis de rendimiento de campañas por día
        - ¿Cuál es el CPC promedio por campaña?
        """)
        
        # Botón para limpiar la conversación en la barra lateral
        if st.button("Limpiar conversación"):
            st.session_state.messages = [
                {"role": "assistant", "content": "¡Hola! Soy tu asistente de análisis de datos. Puedes hacerme cualquier pregunta sobre tus datos de marketing y te ayudaré a encontrar la información que necesitas."}
            ]
            st.rerun()
    
    # Crear un contenedor para los mensajes con estilo fixed height para permitir scrolling
    chat_container = st.container()
    
    # Mostrar historial de mensajes en el contenedor
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message.get("is_dataframe", False):
                    if isinstance(message["content"], pd.DataFrame):
                        st.dataframe(message["content"], use_container_width=True)
                        
                        # Agregar botón de descarga si es un DataFrame
                        csv = message["content"].to_csv(index=False)
                        st.download_button(
                            label="Descargar como CSV",
                            data=csv,
                            file_name="resultados_consulta.csv",
                            mime="text/csv",
                            key=f"download_csv_button_general_query_{id(csv)}" 
                        )
                    else:
                        # Si no es un DataFrame pero tiene la marca is_dataframe
                        try:
                            df = pd.DataFrame(message["content"])
                            st.dataframe(df, use_container_width=True)
                        except:
                            st.warning("No se pudo mostrar los datos en formato tabla")
                elif message.get("is_code", False):
                    st.code(message["content"], language="sql")
                elif message.get("is_error", False):
                    st.error(message["content"])
                else:
                    st.markdown(message["content"])
    
    # Entrada de la pregunta usando chat_input
    if question := st.chat_input("Escribe tu pregunta aquí..."):
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Recargar para mostrar mensaje del usuario
        st.rerun()
    
    # Procesar la pregunta del usuario después de la recarga
    # Esto asegura que el mensaje del usuario se muestre antes de procesar
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and not st.session_state.get("processed_last_question", False):
        question = st.session_state.messages[-1]["content"]
        
        # Marcar como procesado para evitar procesarlo múltiples veces
        st.session_state.processed_last_question = True
        
        # Respuesta del asistente
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Procesando tu consulta...")
            
            start_time = time.time()
            
            try:
                # Usar el agente de LangChain
                result = db_agent.execute_query(question)
                
                # Lista para acumular componentes de la respuesta
                response_components = []
                
                # Mostrar resultados
                if result["success"]:
                    # Formatear un mensaje de respuesta principal
                    if "data" in result and isinstance(result["data"], pd.DataFrame) and not result["data"].empty:
                        rows = len(result["data"])
                        cols = len(result["data"].columns)
                        main_response = f"Encontré {rows} resultados para tu consulta. Consulta procesada en {time.time() - start_time:.2f} segundos."
                    else:
                        main_response = f"He procesado tu consulta en {time.time() - start_time:.2f} segundos."
                    
                    # Actualizar el placeholder con el mensaje principal
                    message_placeholder.markdown(main_response)
                    
                    # Agregar el mensaje principal al historial
                    st.session_state.messages.append({"role": "assistant", "content": main_response})
                    response_components.append(main_response)
                    
                    # Mostrar la consulta generada en un expander
                    with st.expander("Ver consulta generada", expanded=False):
                        st.code(result["sql_query"], language="sql")
                    
                    # No guardamos la consulta en el historial para mantener la conversación más limpia
                    
                    # Mostrar tablas relevantes si están disponibles
                    if "relevant_tables" in result:
                        tables_message = f"Tablas consultadas: {', '.join(result['relevant_tables'][:5])}"
                        st.caption(tables_message)
                        # No guardamos esto en el historial
                    
                    # Mostrar resultados como DataFrame en un nuevo mensaje
                    if "data" in result and isinstance(result["data"], pd.DataFrame) and not result["data"].empty:
                        st.write("Aquí están los resultados:")
                        st.dataframe(result["data"], use_container_width=True)
                        
                        # Agregar botón de descarga
                        csv = result["data"].to_csv(index=False)
                        st.download_button(
                            label="Descargar datos como CSV",
                            data=csv,
                            file_name="resultados_consulta.csv",
                            mime="text/csv",
                        )
                        
                        # Agregar el DataFrame al historial de mensajes en un nuevo mensaje
                        st.session_state.messages.append({"role": "assistant", "content": result["data"], "is_dataframe": True})
                    elif "data" in result and isinstance(result["data"], pd.DataFrame) and result["data"].empty:
                        empty_message = "No encontré datos que coincidan con tu consulta."
                        st.write(empty_message)
                        st.session_state.messages.append({"role": "assistant", "content": empty_message})
                        response_components.append(empty_message)
                    
                    # Mostrar análisis si está disponible
                    if "analysis" in result and result["analysis"]:
                        analysis_message = f"**Análisis**: {result['analysis']}"
                        st.write(analysis_message)
                        st.session_state.messages.append({"role": "assistant", "content": analysis_message})
                        response_components.append(analysis_message)
                    
                    # Añadir un mensaje para indicar que pueden hacer otra pregunta
                    follow_up = "¿Hay algo más que te gustaría saber sobre estos datos?"
                    st.write(follow_up)
                    st.session_state.messages.append({"role": "assistant", "content": follow_up})
                    response_components.append(follow_up)
                    
                    # Mostrar métricas de rendimiento en modo debug
                    if debug_mode and "timing" in result:
                        with st.expander("Detalles técnicos", expanded=False):
                            st.json(result["timing"])
                else:
                    # Mostrar mensaje de error
                    error_message = f"Lo siento, no pude procesar tu consulta: {result['error']}"
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message, "is_error": True})
                    
                    # Mostrar detalles técnicos del error
                    if "sql_query" in result:
                        with st.expander("Detalles técnicos del error", expanded=False):
                            st.code(result["sql_query"], language="sql")
                    
                    # Sugerir una reformulación
                    suggestion = "¿Podrías intentar reformular tu pregunta? Por ejemplo, sé más específico sobre qué datos quieres ver."
                    st.write(suggestion)
                    st.session_state.messages.append({"role": "assistant", "content": suggestion})
            
            except Exception as e:
                # Manejar cualquier error inesperado
                error_message = f"Lo siento, ocurrió un error inesperado: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "is_error": True})
                st.session_state.messages.append({"role": "assistant", "content": "Por favor, intenta con otra pregunta o contacta al administrador del sistema si el problema persiste."})
        
        # Desmarcar como procesado para la próxima pregunta
        st.session_state.processed_last_question = False
        
        # Recargar para mostrar las respuestas
        st.rerun()