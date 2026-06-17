from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file,filter_to_minimal_docs,text_split,download_embeddings

load_dotenv()
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

extracted_data=load_pdf_file(data='data/')
filter_data=filter_to_minimal_docs(extracted_data)
texts_chunks=text_split(filter_data)
emdeddings= download_embeddings()

from pinecone import Pinecone
pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

from pinecone import ServerlessSpec
index_name="medicalbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec= ServerlessSpec(cloud='aws',region='us-east-1')
    )

index=pc.Index(index_name)

doc_search=PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=emdeddings,
    index_name=index_name
)

