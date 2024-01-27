from sentence_transformers import SentenceTransformer
import pinecone
import streamlit as st
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
model = SentenceTransformer('all-MiniLM-L6-v2')

# pinecone.init(api_key='', environment='us-east-1-aws')
# index = pinecone.Index('langchain-chatbot')

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# Access Qdrant API information
api_key_qdrant = config['Qdrant']['api_key']
url_qdrant = config['Qdrant']['url']

qdrant_client = QdrantClient(
    url=url_qdrant, 
    api_key=api_key_qdrant,
)


collection_name = "dslogic"
def find_match(input):
    input_em = model.encode(input).tolist()
    results = qdrant_client.search(collection_name=collection_name, query_vector=input_em, limit=2, with_payload=True)
    return "\n".join(point.payload['page_content'] for point in results)

# def query_refiner(conversation, query):

#     response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#     temperature=0.7,
#     max_tokens=256,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string