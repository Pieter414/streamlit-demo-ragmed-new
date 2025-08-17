import streamlit as st
import os
import asyncio
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document, HumanMessage, SystemMessage

import operator, json

# Define the state schema for LangGraph
class ChatState(BaseModel):
    messages: List[Dict[str, Any]]

# Use os for configuring api key
secrets = st.secrets['general']
MISTRAL_API_KEY = secrets['MISTRAL_API_KEY']
HF_TOKEN = secrets['HF_TOKEN']
PERSIST_PATH = secrets['PERSIST_PATH']

os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

llm = ChatMistralAI(
    model='mistral-small-latest', 
    api_key=MISTRAL_API_KEY,
    temperature=0,
)

# --- Setup RAG components ---
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY
)
vectorstore = SKLearnVectorStore(
    persist_path=PERSIST_PATH,
    embedding=embeddings,
    serializer="parquet"
)
retriever = vectorstore.as_retriever()

# --- RAG Relevance Grader ---
class RelevanceGrade(BaseModel):
    binary_score: str = Field(..., description="Either 'yes' or 'no'")

llm_json = llm.with_structured_output(RelevanceGrade)

# Define RAG node for LangGraph
### Nodes
def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    relevant = state.get("documents_relevant", True)

    if not relevant:
        context = "Documents were not relevant, answering from prior knowledge.\n\n"
    else:
        context = ""

    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=context + docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    documents_relevant = False
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = result.binary_score  # ✅ Fixed here
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            documents_relevant = True
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    return {
        "documents": filtered_docs,
        "documents_relevant": documents_relevant,
    }


### Edges
def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)

    result = llm_json.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    
    hallucination_grade = "yes"

    if hallucination_grade.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        answer_grade = result.content["binary_score"]

        if answer_grade.lower() == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"

# Retrieval Grader
doc_grader_instructions = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
    "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."
)

doc_grader_prompt = """Here is the retrieved document: 

{document}

Here is the user question: 

{question}

Carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with a single key: binary_score, that is 'yes' or 'no' to indicate relevance."""
  

# rag_prompt = """You are an assistant for question-answering tasks. 

# Here is the context to use to answer the question:

# {context} 

# Think carefully about the above context. 

# Now, review the user question:

# {question}

# Provide an answer to this questions using only the above context. 

# Use three or four sentences maximum and keep the answer concise.

# Answer the question in Indonesian.

# Answer:"""

rag_prompt = """
You are an Autoimmune Disease Assistant designed exclusively for healthcare professionals.

Here is the context to use to answer the question:
{context}

Guidelines:
1. If the practitioner’s query can be truthfully and factually answered using the knowledge base only, respond concisely (3–4 sentences), politely, and professionally in Indonesian.
2. If the answer is not contained in the knowledge base, reply exactly:
   “Saya tidak mengetahui jawaban atas pertanyaan Anda.”
3. Restrict all answers to the scope of autoimmune conditions and clinical practice; do not address non-clinical or patient-directed topics.
4. In case of a conflict between the raw knowledge base and the new knowledge base, prefer the new knowledge base, and within it, the most recent source.
5. The practitioner’s question is in Indonesian; always detect and respond in Indonesian.
6. Do not generate any additional opening or closing remarks—just the answer.

User question:
{question}

Answer:
"""


# Hallucination grader instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""


# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
# Build LangGraph
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    max_retries: int
    loop_step: Annotated[int, operator.add]
    documents_relevant: bool

builder = StateGraph(state_schema=GraphState)

# Define simplified nodes
builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("generate", generate)

# Direct sequence
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade_documents")
builder.add_edge("grade_documents", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# --- Streamlit UI ---
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