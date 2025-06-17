import streamlit as st
import pandas as pd
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------------------
# Carga de API Key
# ------------------------------------------------------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")  # o directamente:
# openai_api_key = "<TU_API_KEY>"


# ------------------------------------------------------------------------------
# Aplicacion Streamlit
# ------------------------------------------------------------------------------
st.title("Pregunta a tus datos CSV con GPT")

st.sidebar.write("Sube un CSV para analizar y preguntar")

# ------------------------------------------------------------------------------
# Carga de CSV
# ------------------------------------------------------------------------------
file = st.file_uploader("Sube tu CSV aquí", type=["csv"])

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

if file is not None:
    df = pd.read_csv(file)
    st.write("Vista previa de tus datos:")
    st.write(df)

    st.write("Haz una pregunta a tus datos:")
    query = st.text_input("Pregunta:")

    if query:
        # Selecciona columnas relevantes según la pregunta
        cols = columnas_relevantes(query, df.columns)
        df_filtrado = df[cols]
        # Limita el número de filas para evitar exceso de tokens
        docs = [f"Registro {i}: " + row.to_json() for i, row in df_filtrado.head(100).iterrows()]
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
        st.write("Respuesta:")
        st.write(response)