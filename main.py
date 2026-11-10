from tool import search
import csv
from itertools import islice
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import ToolMessage

with open("/home/20261109/apiKey-7996169.csv", "r") as f:
    reader = csv.reader(f)
    llm_api_key = next(islice(reader, 1, 2))[1]

model = init_chat_model(
     model='qwen3.8-flash',
            base_url="https://ws-3qlgmyntf26tanke.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            api_key=llm_api_key,
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            model_provider="openai",
            extra_body={
        "enable_thinking": True
    }
)

tools = [search]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个聊天助手，能够调用工具回答用户的问题",
    checkpointer=MemorySaver()
)

def run_agent():
    while True:
        user_input = input("请输入：")
        if user_input.lower() == 'exit':
            break
        response = agent.invoke({'messages': [{'role': 'user', 'content': user_input}]},config={"configurable": {"thread_id": "1"}})
        if response.get('messages')[-1] is ToolMessage:
            response = agent.invoke({'messages': response.get('messages')},config={"configurable": {"thread_id": "1"}})
        print(response.get('messages')[-1].content)

if __name__ == "__main__":
    run_agent()