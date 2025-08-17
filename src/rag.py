# -- IMPORT --

# state scheme import
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, Optional
from typing_extensions import TypedDict

# env import
import os
from dotenv import load_dotenv

# mistral import
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

# langgraph import
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# langchain import
from langchain.chains import RetrievalQA
from langchain.schema import Document, HumanMessage, SystemMessage

# misc import
import operator, json

def setup_graph():
    # -- SETUP SCHEMA STATE --

    # Define the state schema for the pipeline
    class ChatState(BaseModel):
        messages: List[Dict[str, Any]]

    # -- SETUP CONFIGURATION --

    # Configure the secrets keys
    load_dotenv()
    MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
    HF_TOKEN = os.environ.get('HF_TOKEN')
    PERSIST_PATH = '../data/Granulomatis_with_Poliangiitis_updated_database'

    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
    os.environ["HF_TOKEN"] = HF_TOKEN

    # -- SETUP LLM --

    # llm model (generate)
    llm = ChatMistralAI(
        model='mistral-small-latest', 
        api_key=MISTRAL_API_KEY,
        temperature=0,
    )

    # Embedding model setup
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=MISTRAL_API_KEY
    )

    # -- SETUP RETRIEVAL --

    # Setup retrieval (Database exists)
    vectorstore = SKLearnVectorStore(
        persist_path=PERSIST_PATH,
        embedding=embeddings,
        serializer="parquet"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}  # fetch 10 candidates, return 3 diverse ones
    )

    # -- SETUP DOCUMENT RELEVANCE --

    # Setup document relevance & hallucination grader
    class Grader(BaseModel):
        binary_score: str = Field(..., description="Either 'yes' or 'no'") # document relevance
        explanation: Optional[str] = Field(None, description="Explanation for the score") # hallucination


    # -- SETUP PROMPT

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

    # -- SETUP NODE & EDGE --

    ## Nodes

    def retrieve(state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        loop_step = state.get("loop_step", 0)

        # Write retrieved documents to documents key in state
        documents = retriever.invoke(question)
        return {
            "documents": documents,
            "loop_step": loop_step + 1
        }


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

        # If the document not relevant, ignore the document
        if not documents or not relevant:
            context = "Documents were not relevant, answering from prior knowledge.\n\n"
            return {
                "generation": "Maaf, saya tidak menemukan informasi relevan di database.",
            }
        else:
            context = ""

        docs_txt = format_docs(documents)
        rag_prompt_formatted = rag_prompt.format(context=context + docs_txt, question=question)
        generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

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

        elif loop_step <= max_retries:
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

        elif loop_step <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    ## -- SETUP GRAPH

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
            "max retries": END,
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