# ==============================================================================
# IMPORTS Y CONFIGURACIÓN INICIAL
# ==============================================================================
import streamlit as st  # Framework para apps web interactivas en Python
import pandas as pd  # Manipulación de datos
import os  # Acceso a variables de entorno y sistema
import matplotlib.pyplot as plt  # Visualización de datos

from langchain.text_splitter import CharacterTextSplitter  # División de texto para LLM
from langchain_openai import OpenAIEmbeddings  # Embeddings de OpenAI
from langchain_community.vectorstores import FAISS  # Almacenamiento vectorial
# --- LCEL pipeline ---
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_community.chat_models import ChatOpenAI  # Modelos de chat de OpenAI
from dotenv import load_dotenv  # Carga variables de entorno desde .env
import openai  # Cliente OpenAI
import traceback  # Manejo de errores y trazas

load_dotenv()  # Cargar variables de entorno (como la API Key)

# ------------------------------------------------------------------------------
# Carga de API Key desde variable de entorno
# ------------------------------------------------------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")  # o directamente:
# openai_api_key = "<TU_API_KEY>"

# ------------------------------------------------------------------------------
# Configuración de la aplicación Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat with CSV",  # Título de la pestaña
    page_icon=":bar_chart:",    # Icono de la pestaña
    layout="centered",          # Layout centrado
)
st.title("Chat with your CSV 📊")  # Título principal de la app

# ------------------------------------------------------------------------------
# Carga de CSV
# ------------------------------------------------------------------------------

st.badge("Sube tu archivo CSV", color="red")  # Badge de aviso
file = st.file_uploader("Sube tu csv", type=["csv"], label_visibility="collapsed")  # Subida de archivo

# ------------------------------------------------------------------------------
# Función para filtrar columnas relevantes usando LLM
# ------------------------------------------------------------------------------
def filter_columns_llm(query, columns):
    """
    Dada una pregunta del usuario y las columnas del dataset,
    usa un LLM para seleccionar solo las columnas relevantes para responder la pregunta.
    """
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
        max_tokens=150,
        temperature=0.1
    )
    content = response.choices[0].message.content.strip()
    cols = [col.strip() for col in content.split(",") if col.strip() in columns]
    if not cols:
        cols = list(columns)
    return cols

# ------------------------------------------------------------------------------
# Función para clasificar la pregunta del usuario (TEXTO o GRAFICA)
# ------------------------------------------------------------------------------
def classify_query(query):
    """Clasifica la pregunta del usuario para determinar el tipo de consulta: TEXTO o GRAFICA."""
    print("Clasificando la pregunta:", query)
    prompt = (
        f"""
        Dada la siguiente pregunta de usuario sobre un dataset en CSV, tienes que clasificar la pregunta en dos categorías:

            1- TEXTO: Pregunta de texto: El usuario solicita información sobre los datos, pero no especifica que sean en una grafica.
            2- GRAFICA: Pregunta de gráfica: El usuario solicita obtener los datos en una grafica.

        NO digas grafica si la pregunta no lo pide explícitamente. Por defecto clasifica como TEXTO.
        Solo devuelve GRAFICA si el usuario pide explicitamente que sea una grafica, o algun tipo concreto de grafica. Devuelve solo la categoría, sin explicación.
        En caso de duda, clasifica la pregunta como TEXTO.
        SI el usuario ESCRIBE grafica, grafico, grafica de barras, grafica de lineas, grafica de pastel, histograma, o cualquier tipo de grafica, entonces clasifica como GRAFICA.
        
        Estos son algunos ejemplos de preguntas:

            Pregunta: ¿Qué trabajos hay en Madrid?
            Respuesta: TEXTO

            Pregunta: Dame un histograma de los trabajos por ubicación.
            Respuesta: GRAFICA

            Pregunta: {query}
            Respuesta:

        """.format(query=query)
    )

    print("Prompt para clasificación:", prompt)

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en análisis de datos. Tu tarea es clasificar la pregunta del usuario para decidir si la respuesta requiere una grafica o no. No des explicaciones, solo devuelve la categoría."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    result = response.choices[0].message.content.strip()
    print("✅ Clasificación de la pregunta:", result)
    return result.upper() if result else "TEXTO"

# ------------------------------------------------------------------------------
# Función para generar y mostrar una gráfica relevante usando IA
# ------------------------------------------------------------------------------
def plot_graph(df_filtrado, query):
    """
    Genera el código Python necesario para crear una gráfica relevante según la pregunta del usuario,
    ejecuta el código y muestra la gráfica en Streamlit.
    """
    code = None
    # Pedir al modelo que genere el código Python para la gráfica
    prompt_code = f"""
        Eres un experto en análisis de datos con Python y pandas.
        El usuario te pide una gráfica sobre el siguiente DataFrame (df):
        {df_filtrado.to_markdown()}
        Pregunta: '{query}'
        Devuelve solo el código Python necesario para generar la gráfica usando plotly, matplotlib o pandas, y mostrarla con plt.show(). No expliques nada, solo el código. El DataFrame ya está cargado como 'df_filtrado'. Ejemplo de respuesta:
        
        import matplotlib.pyplot as plt
        df_filtrado['columna'].value_counts().plot.pie()
        plt.show()

        Haz las graficas simples, pero vistosas, con colores agradables pastel y etiquetas claras. Si puedes, que sean interactivas, y ten en cuenta que no se solapen los textos de las etiquetas. Ajustalo para que se vea simple, bonito y limpio. 
        """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response_code = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en Python y visualización de datos."},
            {"role": "user", "content": prompt_code}
        ],
        max_tokens=1000,
        temperature=0.2
    )
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

# ------------------------------------------------------------------------------
# Función para combinar múltiples respuestas de la IA en una sola
# ------------------------------------------------------------------------------
def combine_responses(responses, query):
    """
    Combina varias respuestas generadas por el modelo LLM en una sola respuesta final,
    resumiendo y seleccionando lo más relevante.
    """
    prompt = f"""
        Eres un asistente experto en análisis de datos. Combina las siguientes respuestas de un modelo llm en una sola, ofreciendo un resumen claro y conciso, teniendo en cuenta la pregunta
            Pregunta: {query}
            Respuestas: {responses}

        Devuelve una respuesta final que sea clara, concisa y fácil de entender. No repitas las respuestas, solo ofrece una respuesta final que combine lo más relevante de todas las respuestas.
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ------------------------------------------------------------------------------
# Función para responder preguntas de texto sobre el DataFrame
# ------------------------------------------------------------------------------
def text_answer(df_filtrado, query):
    """
    Divide el DataFrame en partes pequeñas y pregunta a la IA por cada parte,
    luego combina las respuestas en una sola respuesta final y la muestra en Streamlit.
    """
    response = None
    if not response:
        status_placeholder = st.empty()
        progress_bar = st.progress(0.0)
    # Nueva lógica: dividir el DataFrame en partes pequeñas (por ejemplo, de 66 en 66 filas)
    chunk_size = 65
    respuestas = []
    total = len(df_filtrado)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        status_placeholder.info(f"Procesando chunk {start+1} a {end} de {total} registros")
        progress_bar.progress(min((end) / total, 1.0))
        sub_df = df_filtrado.iloc[start:end]
        docs = [f"Registro {i+start}: " + row.to_json() for i, row in sub_df.iterrows()]
        context = "\n".join(docs)
        prompt = f"""
        Eres un asistente experto en análisis de datos. Responde de forma clara, pero dando explicaciones amables. Razona con detenimiento las respuestas. Siempre ofrece los resultados formateados, y lo más bonito posible. Sugiere preguntas de seguimiento interesantes sobre los datos, para obtener los insights más relevantes.\n
        Pregunta: {query}\nContexto:\n{context}
        """
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en análisis de datos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            content = response.choices[0].message.content.strip()
            respuestas.append(content)
        except Exception as e:
            respuestas.append(f"[Error procesando chunk: {e}]")
    final_answer = combine_responses(respuestas, query)
    status_placeholder.empty()
    progress_bar.empty()
    st.markdown(f"""
        <div style='background: #fff; border-radius: 12px; padding: 20px; margin-top: 20px; box-shadow: 8px 8px 0px #1c1c1c; border: 2px solid #1c1c1c;'>{final_answer}
        </div>
    """, unsafe_allow_html=True) 

# ------------------------------------------------------------------------------
# Lógica principal de la app: carga de archivo, pregunta y respuesta
# ------------------------------------------------------------------------------
if file is not None:
    df = pd.read_csv(file, delimiter=";", encoding="utf-8")  # Cargar CSV
    st.write("Vista previa de tus datos")
    st.write(df)

    st.write("Haz una pregunta a tus datos:")
    if 'user_query' not in st.session_state:
        st.session_state['user_query'] = ''
    query = st.text_input("Pregunta", value=st.session_state['user_query'], key="input_query", label_visibility="collapsed", placeholder="Ej: ¿Qué trabajos hay en Madrid?")

    if query:
        # Selecciona columnas relevantes según la pregunta
        with st.spinner('Analizando el dataset... 🚀'):
            cols = filter_columns_llm(query, df.columns)
            df_filtrado = df[cols]

            query_classification = classify_query(query)

            if query_classification == "GRAFICA":
                plot_graph(df_filtrado, query)
            else:
                text_answer(df_filtrado, query)

