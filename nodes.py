import os
from typing import Optional, List

from langchain_core.tools import BaseTool
from langgraph.constants import END
from langgraph.graph import MessagesState
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from base import get_tools, get_tools_by_name
from schemas import State
from tools import write_an_article, schedule_date_for_article, check_daily_number_article

# Token e configurazione modello
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

tools = get_tools()
tools_by_name = get_tools_by_name(tools)

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

llm = ChatHuggingFace(llm=llm_endpoint)
llm_with_tools = llm.bind_tools([write_an_article], tool_choice="any")

# Il nodo usa direttamente l'llm definito sopra
def call_llm(state: MessagesState):
    risposta = llm_with_tools.invoke(state["messages"])
    return {"messages": [risposta]}

"""Significato: Questo dice all'LLM chi ha generato questo messaggio. Nel mondo dei modelli di chat ci sono 
quattro ruoli standard: "system" (le regole), "user" (l'utente umano), "assistant" (l'LLM stesso) e "tool" 
(le funzioni Python). Scrivendo "tool", l'LLM capisce che questo testo non è una nuova richiesta dell'utente, 
ma il risultato del compito che lui stesso aveva richiesto."""

def tool_node(state: State):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
            else:
                return "action"



"""
state = {
    "messages": [
        ... ovvero i messaggi precedenti ...,
        {
            "content": "", 
            "tool_calls": [
                {
                    "name": "write_email",
                    "args": {"to": "luca@test.com", "text": "Ciao"},
                    "id": "call_123"
                }
            ]
        }
    ]
}
"""

