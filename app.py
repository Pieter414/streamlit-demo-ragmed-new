import streamlit as st
import os
import asyncio
from src import rag
import time

# --- Setup Graph ---

# Timer
start_time = time.time()
time_placeholder = st.empty()

# Setup graph
graph = rag.setup_graph()

# --- Streamlit UI ---

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
    
    last_output = None  # store only the latest generation
    
    async for step in graph.astream(input_state, stream_mode="values"):
        if "generation" in step:
            last_output = (
                step["generation"].content 
                if hasattr(step["generation"], "content") 
                else step["generation"]
            )
   
    total_time = time.time() - start_time
    st.chat_message("assistant").write(f"✅ Done in {total_time:.1f} seconds")
            
    # write only the final generation after loop ends
    if last_output:
        st.chat_message("assistant").write(last_output)
        st.session_state.history.append({"role": "assistant", "content": last_output})

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