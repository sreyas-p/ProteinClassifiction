from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
import time
import logging
from collections import deque
from typing import Dict, List
import importlib
import openai
import chromadb
import tiktoken as tiktoken
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import re
import weaviate
from weaviate.auth import AuthApiKey

client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WCD_DEMO_URL"),
    auth_credentials=AuthApiKey(api_key=os.getenv("WCD_DEMO_RO_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")  # <-- Replace with your API key
    }
)

collection_name = "GitBookChunk"

chunks = client.collections.get(collection_name)
response = chunks.generate.near_text(
    query="Gene Sequence",
    limit=3,
    grouped_task="generate gene ontology annotations for each protien given the sequence and structure of it"
)

print(response.generated)

chunks_list = list()
for i, chunk in enumerate(chunked_text):
    data_properties = {
        "chapter_title": "What is Git",
        "chunk": chunk,
        "chunk_index": i
    }
    data_object = wvc.data.DataObject(properties=data_properties)
    chunks_list.append(data_object)
chunks.data.insert_many(chunks_list)    

response = chunks.aggregate.over_all(total_count=True)
print(response.total_count)