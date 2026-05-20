from langgraph.graph import StateGraph, MessagesState, START, END
# Importiamo la funzione del nodo dal file nodes.py
from nodes import call_llm, tool_node, should_continue


workflow = StateGraph(MessagesState)

# Aggiungiamo il nodo importato
workflow.add_node("call_llm", call_llm)
workflow.add_node("call_tool", tool_node)

workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_continue,{
    "action": "call_tool",
    END: END,
  },
)
workflow.add_edge("call_llm", "call_tool")

# Compilazione pulita senza argomenti extra
app = workflow.compile()

"""if __name__ == "__main__":
    print("Inviando la domanda al grafo locale...")

    output = app.invoke({"messages": [{"role": "user", "content": "Ciao! Dimmi solo 'Funziona!' se mi senti."}]})

    print("\nRisposta dell'Agente:")
    print(output["messages"][-1].content)"""