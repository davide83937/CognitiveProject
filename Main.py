from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

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
    # IMPORTANTE: Cambiamo il thread_id in "sessione_5" per partire con
    # una memoria pulita e non portarci dietro i crash precedenti
    config = {"configurable": {"thread_id": "sessione_8"}}

    while True:
        print("Inviando la domanda al grafo locale...")
        content = input()

        # 1. Chiediamo al grafo in che stato si trova attualmente
        state = topic_assistant.get_state(config)

        # 2. Se 'next' non è vuoto, significa che il grafo è freezato sull'interrupt()
        if state.next:
            print("⚙️ Risveglio il grafo in pausa e invio il tuo feedback...")
            # Usiamo Command(resume=...) per sbloccare l'interrupt e passargli la tua stringa
            output = topic_assistant.invoke(Command(resume=content), config=config)
        else:
            # Se è vuoto, il grafo ha finito ed è pronto per una nuova richiesta normale
            output = topic_assistant.invoke({
                "messages": [{"role": "user", "content": content}],
                "prompt_input": content
            }, config=config)

        print("\nRisposta dell'Agente:")
        # Piccolo controllo di sicurezza prima di stampare l'ultimo messaggio
        if output and "messages" in output and len(output["messages"]) > 0:
            print(output["messages"][-1].content)