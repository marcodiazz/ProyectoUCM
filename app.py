import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import openai
load_dotenv()

# ------------------------------------------------------------------------------
# Carga de API Key
# ------------------------------------------------------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")  # o directamente:
# openai_api_key = "<TU_API_KEY>"


# ------------------------------------------------------------------------------
# Aplicacion Streamlit
# ------------------------------------------------------------------------------
st.title("Pregunta a tu CSV con IA 📊")

# st.sidebar.write("Sube un CSV para analizar y preguntar")

# ------------------------------------------------------------------------------
# Carga de CSV
# ------------------------------------------------------------------------------
st.badge("Sube tu archivo CSV", color="red")
file = st.file_uploader("Sube tu csv", type=["csv"], label_visibility="collapsed")

def columnas_relevantes(pregunta, columnas):
    pregunta = pregunta.lower()
    # Mapas de columnas según palabras clave
    if "descripción" in pregunta or "detalle" in pregunta:
        return [col for col in columnas if col in ["title", "company", "description", "location"]]
    elif "salario" in pregunta:
        return [col for col in columnas if col in ["title", "company", "salary", "location"]]
    elif "ubicación" in pregunta or "ciudad" in pregunta or "donde" in pregunta:
        return [col for col in columnas if col in ["title", "company", "location"]]
    elif "tecnología" in pregunta or "stack" in pregunta:
        return [col for col in columnas if col in ["title", "company", "technologies"]]
    elif "idioma" in pregunta or "languages" in pregunta:
        return [col for col in columnas if col in ["title", "company", "languages"]]
    elif "contrato" in pregunta:
        return [col for col in columnas if col in ["title", "company", "contract_type"]]
    else:
        # Por defecto, columnas principales sin descripción
        return [col for col in columnas if col in ["title", "company", "location"]]

def filter_columns_llm(query, columns):
    print("Filtrando columnas relevantes para la pregunta:", query)
    print("Columnas disponibles:", columns)
    prompt = (
        """
        Dada la siguiente pregunta de usuario sobre un dataset en CSV, 
        Elige solo las columnas relevantes de la lista proporcionada. 
        Devuelve una lista de nombres de columnas exactos, separados por coma, sin explicación.
        En caso de duda, no elijas la columna "description".
        
        Pregunta: {pregunta}
        Columnas disponibles: {columnas}\n
        Columnas relevantes: []
        """.format(pregunta=query, columnas=", ".join(columns))
    )
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en análisis de datos. Tu tarea es seleccionar las columnas más relevantes de un dataset CSV para responder a preguntas del usuario. No des explicaciones, solo devuelve los nombres de las columnas."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    cols = [col.strip() for col in content.split(",") if col.strip() in columns]
    if not cols:
        cols = list(columns)
    return cols


if file is not None:
    df = pd.read_csv(file, delimiter=";", encoding="utf-8")
    st.write("Vista previa de tus datos")
    st.write(df)

    st.write("Haz una pregunta a tus datos:")
    if 'user_query' not in st.session_state:
        st.session_state['user_query'] = ''
    query = st.text_input("Pregunta", value=st.session_state['user_query'], key="input_query", label_visibility="collapsed", placeholder="Ej: ¿Qué trabajos hay en Madrid?")

    if query:
        st.session_state['user_query'] = ''  # Limpiar input tras enviar
        # Selecciona columnas relevantes según la pregunta
        # cols = columnas_relevantes(query, df.columns)
        with st.spinner('Analizando el dataset... 🚀'):
            cols = filter_columns_llm(query, df.columns)
            df_filtrado = df[cols]
            # Limita el número de filas para evitar exceso de tokens
            docs = [f"Registro {i}: " + row.to_json() for i, row in df_filtrado.head(len(df_filtrado)).iterrows()]
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            chunks = text_splitter.split_text("\n".join(docs))

            embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.from_texts(chunks, embedding)
            retriever = db.as_retriever()
            model = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
            qa = RetrievalQA.from_chain_type(
                llm=model,
                retriever=retriever,
                return_source_documents=False
            )
            response = qa.run(query)
        st.markdown(f"""
            <div style='background: #fff; border-radius: 12px; padding: 20px; margin-top: 20px; box-shadow: 8px 8px 0px #e3e8ee; border: 2px solid #5f5dff;'>
                <b>Respuesta</b><br>{response}
            </div>
        """, unsafe_allow_html=True)

        # --- Nueva funcionalidad: generación dinámica de código de gráficas ---
        grafica_keywords = ["gráfica", "grafica", "plot", "histograma", "distribución", "distribucion", "gráfico", "grafico", "visualiza", "visualización", "visualizacion", "tarta", "pie"]
        if any(k in query.lower() for k in grafica_keywords):
            st.info("Generando gráfica relevante con IA...")
            # Pedir al modelo que genere el código Python para la gráfica
            prompt_code = f"""
                Eres un experto en análisis de datos con Python y pandas. El usuario te pide una gráfica sobre el siguiente DataFrame (df):\n\n{df_filtrado.head(10).to_markdown()}\n\nPregunta: '{query}'\n\nDevuelve solo el código Python necesario para generar la gráfica usando matplotlib o pandas, y mostrarla con plt.show(). No expliques nada, solo el código. El DataFrame ya está cargado como 'df_filtrado'.\n\nEjemplo de respuesta:\nimport matplotlib.pyplot as plt\ndf_filtrado['columna'].value_counts().plot.pie()\nplt.show()\n"""
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response_code = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en Python y visualización de datos."},
                    {"role": "user", "content": prompt_code}
                ],
                max_tokens=300,
                temperature=0
            )
            import traceback
            code = response_code.choices[0].message.content.strip()
            # Eliminar bloques de markdown (```python, ```, etc.)
            code = code.replace('```python', '').replace('```', '').strip()
            code = code.replace('df[', 'df_filtrado[').replace('df.', 'df_filtrado.')
            # Eliminar importaciones innecesarias y comentarios
            code_lines = [line for line in code.splitlines() if 'import matplotlib' not in line and not line.strip().startswith('#')]
            # Eliminar líneas vacías al inicio y final
            while code_lines and code_lines[0].strip() == '':
                code_lines.pop(0)
            while code_lines and code_lines[-1].strip() == '':
                code_lines.pop()
            code = '\n'.join(code_lines)
            try:

                plt.close('all')  # Cierra figuras previas
                fig = plt.figure()
                # Ejecutar el código generado
                exec(code, {"plt": plt, "df_filtrado": df_filtrado, "pd": pd})
                # Captura la figura activa
                st.pyplot(plt.gcf())
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"Error al generar la gráfica: {e}\n\nCódigo generado:\n{code}\n\nTraceback:\n{tb}")