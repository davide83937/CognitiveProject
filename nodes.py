import os
from langgraph.graph import MessagesState
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from tools import write_description

# Token e configurazione modello
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)
llm = ChatHuggingFace(llm=llm_endpoint)
llm_with_tools = llm.bind_tools([write_description], tool_choice="any")

# Il nodo usa direttamente l'llm definito sopra
def call_llm(state: MessagesState):
    risposta = llm_with_tools.invoke(state["messages"])
    return {"messages": [risposta]}