# -- IMPORT --

# import streamlit keys
import streamlit as st

# state scheme import
from pydantic import BaseModel, Field, SecretStr
from typing import List, Dict, Any, Annotated, Optional
from typing_extensions import TypedDict

# env import
import os
import glob
from dotenv import load_dotenv

# llm langchain import
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI

# langgraph import
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# langchain import
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_core.utils.utils import secret_from_env
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# misc import
import operator, json
import time
from functools import lru_cache
import hashlib

# other py files import
from . import retrieve as r

# -- SETUP CONFIGURATION --

# Configure the secrets keys (Streamlit Version)
secrets = st.secrets['general']
OPENROUTER_API_KEY = secrets['OPENROUTER_API_KEY']
MISTRAL_API_KEY = secrets['MISTRAL_API_KEY']
HF_TOKEN = secrets['HF_TOKEN']
PERSIST_PATH = secrets['PERSIST_PATH']

# Multi Query Retrieval setup
N_PARAPHRASES = 3            # number of paraphrases to generate
MAX_MERGED_DOCS = 5          # final docs to return after dedupe
PARAPHRASE_CACHE_SIZE = 512
RETRIEVAL_CACHE_SIZE = 2048

# -- SETUP SCHEMA STATE --

# Define the state schema for the pipeline
class ChatState(BaseModel):
    messages: List[Dict[str, Any]]

# Define OpenRouter integration with langchain
class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("OPENROUTER_API_KEY", default=None)
    )
    
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(base_url="https://openrouter.ai/api/v1", openai_api_key=openai_api_key, **kwargs)
        
        
# -- SETUP DOCUMENT RELEVANCE --

# Setup document relevance & hallucination grader
class Grader(BaseModel):
    binary_score: str = Field(..., description="Either 'yes' or 'no'") # document relevance
     
        
def setup_graph():
    # -- SETUP LLM --

    # Mistal AI llm model (generate)
    llm = ChatMistralAI(
        model='mistral-small-latest', 
        api_key=MISTRAL_API_KEY,
        temperature=0,
    )
    
    # LLM setup (json)
    llm_json = llm.with_structured_output(Grader)

    # -- SETUP RETRIEVAL --
    retrievers = r.setup_retrieval()
    
    # -- SETUP PROMPT ---

    # Setup prompt
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

    hallucination_grader_instructions = """
    You are a teacher grading a quiz. 

    You will be given FACTS and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of yes means that the student's answer meets all of the criteria. 
    A score of no means that the student's answer does not meet all of the criteria.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
    Avoid simply stating the correct answer at the outset.
    """

    hallucination_grader_prompt = """
    FACTS: 

    {documents}

    STUDENT ANSWER: {generation}

    Return JSON with two keys:
    - binary_score: either 'yes' or 'no'
    - explanation: step-by-step reasoning for the score.
    """

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

    answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

    multivector_router_prompt = """
    You are a medical assistant. Choose the most relevant database (retriever) based on the user's question. 
    Available options are:
    {options_text}

    Question: {question}

    Return ONLY the retriever name (exactly as listed).
    """

    multi_query_paraphrasing_prompt = """
        Buat {n} parafrase singkat (1-2 kalimat) dari pertanyaan pengguna berikut.\n

        Berikan tiap parafrase di baris baru, tanpa nomor:\n\n{query}\n
    """

    # -- SETUP NODE & EDGE --

    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


    def _doc_id(doc):
        """
        Try canonical id, fallback to hash of page_content.
        """
        try:
            mid = getattr(doc, "metadata", None) or {}
            for k in ("id", "source", "doc_id", "source_id"):
                if k in mid and mid[k]:
                    return str(mid[k])
        except Exception:
            pass
        return _hash_text(getattr(doc, "page_content", "") or "")


    @lru_cache(maxsize=PARAPHRASE_CACHE_SIZE)
    def generate_paraphrases_cached(query:str, n:int):
        """
        Returns tuple of paraphrases. lru_cache requires hashable return;
        tuple is immutable and OK to cache.
        """

        prompt = multi_query_paraphrasing_prompt.format(
            n=N_PARAPHRASES, query=query
        )

        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or (resp if isinstance(resp, str) else "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        paraphrases = lines[:n] if len(lines) >= n else lines

        if not paraphrases:
            paraphrases = [query]
        return tuple(paraphrases)


    @lru_cache(maxsize=RETRIEVAL_CACHE_SIZE)
    def cached_retrieve_for_paraphrase(retriever_name: str, paraphrase: str):
        """
        Looks up retriever by name and calls retriever.invoke(paraphrase).
        """
        retriever_info = next(
            (r for r in retrievers if r["name"].lower() == retriever_name.lower()), None
        )

        if retriever_info is None:
            print("⚠️ No retriever found, falling back to Arthritis DB")
            retriever_info = next(r for r in retrievers if r["name"] == "Arthritis")

        retriever = retriever_info["retriever"]

        docs = retriever.invoke(paraphrase)
        return docs


    def route_question(question: str):
        """
        Routing question decision for choosing the right disease's database
        """
        options_text = "\n".join(
            [f"- {r['name']}: {r['description']}" for r in retrievers]
        )

        router_prompt = multivector_router_prompt.format(
            options_text=options_text, question=question
        )

        raw_response = llm.invoke(router_prompt).content.strip()

        # Step 2: Match retriever
        retriever_info = next(
            (r for r in retrievers if r["name"].lower() == raw_response.lower()), None
        )

        if retriever_info is None:
            return "Arthritis"  # fallback

        return retriever_info["name"]


    def multi_query_generation(question, retriever_name):
        """
        Multi Query Paraphases generation + Merge/Dedupe document reranking

        Args:
            question (str): user's question
            retriever_name (str): retriever's database name

        Returns:
            merged (list): Merged and deduped related documents used for retrieval
        """
        # paraphasing
        paraphrases = generate_paraphrases_cached(question, N_PARAPHRASES)
        paraphrases = list(paraphrases)  # convert tuple -> list for printing/iteration
        print(f"Paraphrases: {paraphrases}")

        # retrieve for every paraphase
        all_docs = []
        for p in paraphrases:
            docs = cached_retrieve_for_paraphrase(retriever_name, p)
            if docs:
                # ensure it's extendable list
                try:
                    all_docs.extend(docs)
                except TypeError:
                    # if retriever returns a single Document, wrap it
                    all_docs.append(docs)

        # dedupe while preserving order
        merged = []
        seen = set()
        for d in all_docs:
            did = _doc_id(d)
            if did in seen:
                continue
            seen.add(did)
            merged.append(d)
            if len(merged) >= MAX_MERGED_DOCS:
                break

        print(f"Retrieved {len(merged)} merged docs (after dedupe).")
        return merged


    def retrieve(state):
        """
        Enhanced version of retrieve documents from vectorstore. Using Multi-query paraphrases + merge/dedupe retrieval
        Uses in-memory lru_cache (no disk persistence).

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents and loop step that countered
            each retrieval
        """
        print("---RETRIEVE (multi-query enhanced) ---")
        question = state["question"]
        loop_step = state.get("loop_step", 0)

        # Write retrieved documents to documents key in state
        retrieve_database = route_question(question)
        print(f"Router chose: {retrieve_database}")

        # Match retrieval
        retriever_info = next(
            (r for r in retrievers if r["name"].lower() == retrieve_database.lower()), None
        )
        if retriever_info is None:
            print("⚠️ No retriever found, falling back to Arthritis DB")
            retriever_info = next(r for r in retrievers if r["name"] == "Arthritis")

        retriever_name = retriever_info["name"]

        # Retrieve using Multi Query Retrieval
        documents = multi_query_generation(question, retriever_name)

        return {
            "documents": documents,
            "loop_step": loop_step + 1
        }
    
    
    @lru_cache(maxsize=PARAPHRASE_CACHE_SIZE)
    def cached_generate(question: str, docs_key: str, relevant: bool) -> str:
        """
        Generate using cache
        """
        # If the document not relevant, ignore the document
        if not docs_key or not relevant:
            return "Maaf, saya tidak menemukan informasi relevan di database."

        rag_prompt_formatted = rag_prompt.format(
            context=docs_key,
            question=question
        )
        generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

        return generation.content


    def generate(state):
        """
         Generate answer using RAG on retrieved documents

         Args:
             state (dict): The current graph state

         Returns:
             state (dict): New key added to state, generation, that contains LLM generation
         """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        relevant = state.get("documents_relevant", True)

        docs_txt = format_docs(documents) if documents else ""
        generation = cached_generate(question, docs_txt, relevant)

        return {
            "generation": generation,
        }


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        documents_relevant = False

        # Check every document
        for d in documents:

            doc_grader_prompt_formatted = doc_grader_prompt.format(
                document=d.page_content, question=question
            )

            result = llm_json.invoke(
                [SystemMessage(content=doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )

            # Check if the document relevant or not
            grade = result.binary_score
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


    ## Edges

    def grade_documents_router(state):
        """
        Router to determine if the document relevant or not, this is to make sure that context 
        isn't relevant from the document.

        Args:
            state (dict): The current graph state

        Returns:
            output (str): Output of if the document relevant or not (relevant, retry (if not), and max retries)
        """
        relevant = state.get("documents_relevant", True)
        loop_step = state.get("loop_step", 0)
        max_retries = state.get("max_retries", 3)

        if relevant:
            print("---DOCS RELEVANT, CONTINUE TO GENERATE---")
            return "relevant"

        elif loop_step < max_retries:
            print("---DOCS NOT RELEVANT, RETRY RETRIEVE---")
            return "retry"

        else:
            print("---DOCS NOT RELEVANT, MAX RETRIES---")
            return "max retries"


    def grade_generation_v_documents_and_question(state):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        loop_step = state.get("loop_step", 0)
        max_retries = state.get("max_retries", 3)

        if loop_step > max_retries:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

        hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
            documents=documents, generation=generation
        )

        result = llm_json.invoke(
            [SystemMessage(content=hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        hallucination_grade = result.binary_score

        if hallucination_grade.lower() == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

            answer_grader_prompt_formatted = answer_grader_prompt.format(
                question=question, generation=generation
            )

            result = llm_json.invoke(
                [SystemMessage(content=answer_grader_instructions)]
                + [HumanMessage(content=answer_grader_prompt_formatted)]
            )
            answer_grade = result.binary_score

            if answer_grade.lower() == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] <= max_retries:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"

        elif loop_step < max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    ## -- SETUP GRAPH --

    # Post-processing setup
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Graph setup
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[Document]
        max_retries: int
        loop_step: Annotated[int, operator.add]
        documents_relevant: bool

    # -- SETUP BUILDER

    builder = StateGraph(state_schema=GraphState)

    # Setup node
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("generate", generate)

    # Setup direct sequence
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "grade_documents")

    # Setup conditional sequence
    builder.add_conditional_edges( ## check if the document relevant?
        "grade_documents",
        grade_documents_router,
        {
            "relevant": "generate",
            "retry": "retrieve",
            "max retries": "generate",
        },
    )

    builder.add_conditional_edges( ## check if the answer is hallucination
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "retrieve",
            "useful": END,
            "max retries": END,
            "not useful": "retrieve",
        },
    )

    graph = builder.compile()
    
    return graph