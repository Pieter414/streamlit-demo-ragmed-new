import streamlit as st
import os
import asyncio
from rag import setup_graph
# from pydantic import BaseModel, Field
# from typing import List, Dict, Any, Annotated
# from typing_extensions import TypedDict

# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from langchain_community.vectorstores import SKLearnVectorStore
# from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.schema import Document, HumanMessage, SystemMessage

# --- Streamlit UI ---

graph = setup_graph()

st.set_page_config(page_title="QA Autoimmune", page_icon="🤖", layout="wide")
st.title("📚 RAG Autoimmune")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Question:")
    submitted = st.form_submit_button("Send")

# --- Async handler ---
async def run_graph(user_input):
    st.session_state.history.append({"role": "user", "content": user_input})
    messages = [{"role": "user", "content": user_input}]
    input_state = {
        "question": user_input,
        "loop_step": 0,
        "max_retries": 3,
    }

    async for step in graph.astream(input_state, stream_mode="values"):
        if "generation" in step:
            output = step["generation"].content if hasattr(step["generation"], "content") else step["generation"]
            st.chat_message("assistant").write(output)
            st.session_state.history.append({"role": "assistant", "content": output})

# Handle submission
if submitted and user_input:
    st.chat_message("user").write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    assistant_message = asyncio.run(run_graph(user_input))
    st.session_state.history.append({"role": "assistant", "content": assistant_message})

# # Render existing history
# for chat in st.session_state.history:
#     st.chat_message(chat["role"]).write(chat["content"])

# Footer
st.markdown("---")
st.markdown("RAG Test Production 🚀")

## --- ARCHIVE ---
# st.set_page_config(page_title="QA Autoimmune", page_icon="🤖", layout="wide")

# # Title
# st.markdown("<h1 style='text-align:center;'>📚 RAG Autoimmune</h1>", unsafe_allow_html=True)

# # Initialize session state for chat
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Chat container
# chat_container = st.container()

# # Input at bottom
# input_col, _ = st.columns([4,1])
# with input_col:
#     user_input = st.text_input("", placeholder="Type your message...", key="input_box")

# # Send button
# send_button = st.button("Send")

# # Function to render messages
# def render_chat():
#     chat_container.empty()
#     with chat_container:
#         for msg in st.session_state.history:
#             if msg['role'] == 'user':
#                 st.chat_message("user").write(msg['content'])
#             else:
#                 st.chat_message("assistant").write(msg['content'])

# # Handle sending
# if send_button and user_input:
#     # Append user message
#     st.session_state.history.append({"role": "user", "content": user_input})
#     render_chat()

#     # Async call to RAG graph
#     async def run_graph(user_input):
#         messages = [{"role": "user", "content": user_input}]
#         input_state = {"question": user_input, "loop_step": 0, "max_retries": 3}

#         async for step in graph.astream(input_state, stream_mode="values"):
#             if "generation" in step:
#                 output = step["generation"].content if hasattr(step["generation"], "content") else step["generation"]
#                 st.session_state.history.append({"role": "assistant", "content": output})
#                 render_chat()

#     # Run and render
#     asyncio.run(run_graph(user_input))

# # Initial render
# render_chat()