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

def setup_retrieval():
    
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

    # Embedding model setup
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=MISTRAL_API_KEY
    )

    cache_dir = LocalFileStore(f"{PERSIST_PATH}/embedding_cache")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=cache_dir,
        key_encoder=lambda x: hashlib.sha256(x).hexdigest(),
    )
    
    # -- SETUP VECTORSTORES --

    gwp_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/Granulomatis_with_Poliangiitis_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    jia_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/Juvenile_Idiopathic_Arthritis_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    ss_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/Sindrom_Sjrongen_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    sle_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/SLE_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    ra_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/RA_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    psa_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/PsA_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    arthritis_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/Arthritis_new_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    as_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/Ankyolising Spondilitis_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    spon_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/Spondyloarthritis_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    ssc_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/SSc_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )

    vas_vectorstore = SKLearnVectorStore(
        persist_path=f'{PERSIST_PATH}/Vasculitis_updated_database.parquet',
        embedding=cached_embeddings,
        serializer="parquet"
    )
    
    # -- SETUP RETRIEVAL --

    arthritis_retriever = arthritis_vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
    )

    retrievers = [
        {
            "name": "RA",
            "description": "Information about rheumatoid arthritis (RA).",
            "retriever": ra_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "PsA",
            "description": "Information about psoriatic arthritis (PsA).",
            "retriever": psa_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "Arthritis",
            "description": "General information about arthritis.",
            "retriever": arthritis_retriever,
        },
        {
            "name": "Granulomatis with Poliangiitis",
            "description": "Information about Granulomatis with Poliangiitis (GwP).",
            "retriever": gwp_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "Juvenile Idiopathic Arthritis",
            "description": "General information about Juvenile Idiopathic Arthritis (JIA).",
            "retriever": jia_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "Sindrom Sjrongen",
            "description": "General information about Sindrom Sjrongen (SS).",
            "retriever": ss_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "SLE",
            "description": "General information about Systemic Lupus Erythematosus (SLE).",
            "retriever": sle_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "Ankyolising Spondilitis",
            "description": "General information about Ankyolising Spondilitis (AS).",
            "retriever": as_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "Spondyloarthritis",
            "description": "General information about Spondyloarthritis.",
            "retriever": spon_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "SSc",
            "description": "General information about Systemic Sclerosis (SSc).",
            "retriever": ssc_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
        {
            "name": "Vasculitis",
            "description": "General information about Vasculitis.",
            "retriever": vas_vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
            ),
        },
    ]
    
    return retrievers