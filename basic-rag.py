from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from openai import OpenAI

# Injection Start
# Get the Path of PDF
pdf_path = Path(__file__).parent / "nodejs.pdf"


loader=PyPDFLoader(pdf_path)

# Loader Loads the content from PDF File
docs=loader.load()

#
textSplitter =RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Splits the document in Chunks
splited_docs=textSplitter.split_documents(documents=docs)

embeddings= OpenAIEmbeddings(model="text-embedding-3-large",api_key="sk-")


# Creating a Vector Store Instance
# vectorStore= QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="basic_rag",
#     embedding=embeddings
# )

# Create Embeddings and Storing in Vector Store
# vectorStore.add_documents(documents=splited_docs)
# Injection End


retriver=QdrantVectorStore.from_existing_collection(
     url="http://localhost:6333",
    collection_name="basic_rag",
    embedding=embeddings
)

query="What is nodejs and its file system"

relevant_chunk= retriver.similarity_search(query)



client= OpenAI(api_key="sk-")

System_prompt=f""""
You are an helpfull AI Assitent who responds on the basis of available context

context:
{relevant_chunk}
"""




result= client.chat.completions.create(
    model="gpt-4",
    messages=[
        { "role":"system", "content":System_prompt},
        {"role":"user", "content":query}
        
    ]
)

print(result.choices[0].message.content)





