from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
# Importiamo la funzione del nodo dal file nodes.py
from nodes import call_llm, tool_node, should_continue, triage_prompt, human_review_node, after_human
from schemas import State, StateInput

workflow = StateGraph(MessagesState)

# Aggiungiamo il nodo importato
workflow.add_node("call_llm", call_llm)
workflow.add_node("human_review", human_review_node)
workflow.add_node("call_tool", tool_node)

workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_continue,{
    "action": "human_review",       # Tool pericolosi -> Pausa umana
    "auto_tool": "call_tool",       # Tool sicuri -> Esecuzione immediata
    END: END,
  },
)

# 2. Dopo la tua revisione, si decide: andiamo al tool o torniamo all'LLM?
workflow.add_conditional_edges("human_review", after_human, {
    "call_tool": "call_tool",
    "call_llm": "call_llm"
})

workflow.add_edge("call_tool", "call_llm")
memory = MemorySaver()
# Compilazione pulita senza argomenti extra
app = workflow.compile()

# Build workflow
overall_workflow = (
    StateGraph(State, input_schema=StateInput)
    .add_node(triage_prompt)
    .add_node("response_agent", app)
    .add_edge(START, "triage_prompt")
)

topic_assistant = overall_workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "sessione_4"}}
    while True:
        print("Inviando la domanda al grafo locale...")
        content = input()
        output = topic_assistant.invoke({"messages": [{"role": "user", "content": content}],
    "prompt_input": content},
     config=config)  # <--- Aggiunta della chiave mancante

        print("\nRisposta dell'Agente:")
        print(output["messages"][-1].content)