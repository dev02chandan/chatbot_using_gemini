import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import pandas as pd
import textwrap

# App title and configuration
st.set_page_config(page_title="Maitri AI Chatbot")

# Configure Gemini API
if "GEMINI_API_KEY" in st.secrets:
    gemini_api = st.secrets["GEMINI_API_KEY"]
else:
    gemini_api = st.text_input(
        "Enter Gemini API token:", type="password", key="api_input"
    )
    if gemini_api and gemini_api.startswith("r8_") and len(gemini_api) == 40:
        st.secrets["GEMINI_API_KEY"] = gemini_api
        st.experimental_rerun()

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    hide_sidebar = True
else:
    st.sidebar.write("Please enter your API key")
    hide_sidebar = False

# Hide sidebar if API key is set
if hide_sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Add logo
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

# Load the dataframe with precomputed embeddings
df = pd.read_feather("data_v2_with_embeddings.feather")


# Function to find the best passages
def find_best_passages(query, dataframe, top_n=3):
    query_embedding = genai.embed_content(
        model="models/text-embedding-004", content=query
    )["embedding"]
    dot_products = np.dot(np.stack(dataframe["Embeddings"]), query_embedding)
    top_indices = np.argsort(dot_products)[-top_n:][::-1]
    return dataframe.iloc[top_indices]["Text"].tolist()


# Function to make prompt
def make_prompt(query, relevant_passages):
    escaped_passages = [
        passage.replace("'", "").replace('"', "").replace("\n", " ")
        for passage in relevant_passages
    ]
    joined_passages = "\n\n".join(
        f"PASSAGE {i+1}: {passage}" for i, passage in enumerate(escaped_passages)
    )
    prompt = textwrap.dedent(
        f"""
    Persona: You are Maitri AI Chatbot, representing MaitriAI, a leading software company specializing in web application development, website design, logo design, software development, and cutting-edge AI applications. You are knowledgeable, formal, and detailed in your responses.

    Task: Answer questions about Maitri AI, its services, and related information. Provide detailed and kind responses in a conversational manner.  If the context is relevant to the query, use it to give a comprehensive answer. If the context is not relevant, acknowledge that you do not know the answer. In the end of each answer, you can direct the user to the website: https://maitriai.com/contact-us/, Whatsapp number: 9022049092.

    Format: Respond in a formal and elaborate manner, providing as much relevant information as possible. If you do not know the answer, respond by saying you do not know. The response should be in plain text without any formatting.

    Context: {joined_passages}

    QUESTION: '{query}'
    

    ANSWER:
    """
    )
    return prompt


# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to Maitri AI Chatbot! How can I assist you with information about Maitri AI today?",
        }
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to Maitri AI Chatbot! How can I assist you with information about Maitri AI today?",
        }
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# Function for generating Gemini response
def generate_gemini_response(query):
    relevant_passages = find_best_passages(query, df)
    prompt = make_prompt(query, relevant_passages)
    response = genai.GenerativeModel("models/gemini-1.5-flash-latest").generate_content(
        prompt
    )
    return response.text


# User-provided prompt
if prompt := st.chat_input(disabled=not gemini_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_gemini_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
