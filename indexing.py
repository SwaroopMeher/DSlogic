from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone as pc
from langchain.vectorstores import Pinecone as pcvs
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
import configparser

config = configparser.ConfigParser()
config.read('./.gitignore/config.ini')

# Access Qdrant API information
api_key_qdrant = config['Qdrant']['api_key']
url_qdrant = config['Qdrant']['url']


qdrant_client = QdrantClient(
    url=url_qdrant, 
    api_key=api_key_qdrant,
)

# qdrant_client.create_collection(
#     collection_name="dslogic",
#     vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
# )

data_dir = 'data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(data_dir)

def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


collection_name = "dslogic"
# index = Qdrant.from_documents(docs,
#                                embeddings, 
#                                collection_name=collection_name)

qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url_qdrant,
    prefer_grpc=True,
    api_key=api_key_qdrant,
    collection_name=collection_name,
)

def get_similiar_docs(query,k=3,score=False):
  if score:
    similar_docs = qdrant.similarity_search_with_score(query,k=k)
  else:
    similar_docs = qdrant.similarity_search(query,k=k)
  return similar_docs

