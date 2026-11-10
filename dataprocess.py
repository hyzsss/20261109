from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.documents import Document
import jionlp as jio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import init_embeddings
import csv
from itertools import islice
from langchain_community.vectorstores import Chroma

with open('/home/20261109/apiKey-7996169.csv','r') as f:
    reader = csv.reader(f)
    api_key = next(islice(reader,1,2))[1]

file_path = '/home/20261109/峨眉山讲解词.docx'

loader = UnstructuredWordDocumentLoader(file_path,mode="elements")

doc = loader.load()

doc_list = []
for i in doc:
    doc_list.append(Document(
        page_content=jio.clean_text(i.page_content), 
        metadata=i.metadata
        ))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", " "]
    )

chunks = splitter.split_documents(doc_list)

chunks = [c for c in chunks if len(c.page_content.strip()) > 30]

embeding_model = init_embeddings(
    base_url='https://ws-3qlgmyntf26tanke.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    model="qwen3.7-text-embedding",
    api_key = api_key,
    provider='openai',
    check_embedding_ctx_length=False,
    chunk_size=20
)

db = Chroma.from_documents(
    documents=chunks,
    collection_name='2333',
    embedding=embeding_model,
    persist_directory='/home/chroma_db'
)
