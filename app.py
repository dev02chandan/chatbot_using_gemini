import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import pandas as pd
import textwrap
import chromadb

# App title and configuration
st.set_page_config(page_title="ICICI ETF Knowledge Center Chatbot")

# Set the API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=api_key)

# Add logo
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)
st.title("ICICI ETF Knowledge Center Chatbot")

# Initialize Chroma Client
chroma_client = chromadb.PersistentClient(path="db/")

# Retrieve the collection
try:
    collection = chroma_client.get_collection(name="icici_etf_articles")
except chromadb.errors.InvalidCollectionException:
    print(
        "Collection 'icici_etf_articles' does not exist. Please check your database setup."
    )
    exit()


# Query the Database
def query_database(user_query, n_results=5):
    """Query the Chroma database with a user query."""
    results = collection.query(
        query_texts=[user_query],  # The query string
        n_results=n_results,  # Number of results to retrieve
    )
    return results


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
        Persona: You are the ICICI ETF Chatbot, an expert on Exchange-Traded Funds (ETFs) offered by ICICI. Your role is to assist users in understanding the various ETF products, how to invest, market trends, sector-specific ETFs, and more. You provide insightful, precise, and professional responses to all ETF-related inquiries.

        Task: Answer questions related to ICICI ETFs, their benefits, investment strategies, market performance, and the process of purchasing and managing ETFs. Ensure your responses are informative, concise, and easy to understand. If a query pertains to something outside the scope of ETFs or ICICI's offerings, or the context provided to you, kindly inform the user that you do not have that information.

        Format: Your answers should be clear, professional, and tailored to the needs of both beginner and experienced investors. Provide as much relevant information as possible, explaining technical terms when necessary.s

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
            "role": "model",
            "parts": "Welcome to Maitri AI Chatbot! How can I assist you with information about Maitri AI today?",
        }
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(
        "assistant" if message["role"] == "model" else message["role"]
    ):
        st.write(message["parts"])


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "model",
            "parts": "Welcome to Maitri AI Chatbot! How can I assist you with information about Maitri AI today?",
        }
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# Function for generating Gemini response
def generate_gemini_response(query):
    """Generate a Gemini response to a user query."""

    # First get relevant documents from the database
    results = query_database(query, 5)

    # Print the retrieved documents for verification
    print("===Verification of Retreived Documents: ===\n")
    print("\nDistances: ", results["distances"][0])
    print("\nTitles:", results["metadatas"][0])
    print("\nSnippets: ")
    for i in results["documents"][0]:
        print("\n", i[100:])
    print("\n===End of Verification===")

    # Generate a prompt for the model
    prompt = make_prompt(query, results["documents"][0])

    # Defining Model
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    # Start chat with history form streamlit session state
    chat = model.start_chat(
        history=[
            {"role": msg["role"], "parts": msg["parts"]}
            for msg in st.session_state.messages
        ]
    )

    # Generate response
    response = chat.send_message(prompt)

    return response.text


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "parts": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Initialize response variable
response = ""

# Generate a new response if last message is not from model
if st.session_state.messages[-1]["role"] != "model":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_gemini_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "model", "parts": response}
    st.session_state.messages.append(message)
