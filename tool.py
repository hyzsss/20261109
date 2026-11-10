from langchain.embeddings.base import init_embeddings
import csv
from itertools import islice
from langchain_chroma import Chroma
from langchain.tools import tool
from pydantic import BaseModel, Field

with open('/home/20261109/apiKey-7996169.csv','r') as f:
    reader = csv.reader(f)
    api_key = next(islice(reader,1,2))[1]

embeding_model = init_embeddings(
    base_url='https://ws-3qlgmyntf26tanke.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    model="qwen3.7-text-embedding",
    api_key = api_key,
    provider='openai',
    check_embedding_ctx_length=False,
    chunk_size=20
)

db = Chroma(
    collection_name='2333',
    embedding_function=embeding_model,
    persist_directory='/home/chroma_db'
)

class IputText(BaseModel):
    text:str = Field(description='用来跟向量库进行相似度匹配的问题文本')
    k:int = Field(description='保留前k个相关的资料片段')

@tool(args_schema=IputText)
def search(text:str, k:int):
    '''
    关于峨眉山的资料库，需要回答相关问题时使用
    '''
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(text)
    texts = [c.page_content for c in docs]
    return texts
