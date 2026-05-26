import os
from typing import Literal

from langchain_core.runnables import RunnableLambda
from langgraph.constants import END
from langgraph.graph import MessagesState
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from base import get_tools, get_tools_by_name
from prompts import triage_system_prompt
from schemas import State, HumanFeedbackSchema
from tools import write_an_article, schedule_date_for_article, check_daily_number_article, get_available_dates_in_month

# Token e configurazione modello
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

tools = get_tools()
tools_by_name = get_tools_by_name(tools)

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

llm = ChatHuggingFace(llm=llm_endpoint)
llm_with_tools = llm.bind_tools([write_an_article, schedule_date_for_article, check_daily_number_article,
                                 get_available_dates_in_month])

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage


def call_llm(state: MessagesState):
    # Creiamo una copia pulita dei messaggi per non mandare in crash Hugging Face
    cleaned_messages = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            # Trasformiamo il messaggio del tool in un messaggio dell'utente simulato,
            # così il modello vede la risposta senza crashare sulle specifiche dei tool
            cleaned_messages.append(HumanMessage(content=f"[Risultato del sistema]: {msg.content}"))
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            dettagli_tool = ", ".join([f"{tc['name']} (Dati: {tc['args']})" for tc in msg.tool_calls])
            testo_base = msg.content if msg.content else "Elaborazione in corso..."
            cleaned_messages.append(
                AIMessage(content=f"{testo_base}\n[Azione proposta in precedenza]: {dettagli_tool}"))
        else:
            cleaned_messages.append(msg)
    risposta = llm_with_tools.invoke(cleaned_messages)
    return {"messages": [risposta]}

# Il nodo usa direttamente l'llm definito sopra
"""def call_llm(state: MessagesState):
    risposta = llm_with_tools.invoke(state["messages"])
    return {"messages": [risposta]}"""

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
            elif tool_call["name"] == "check_daily_number_article" or tool_call["name"] == "get_available_dates_in_month":
                # Questa parola chiave salterà il nodo umano
                return "auto_tool"
            else:
                return "action"
    return END

class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""
    # --- IL TRUCCO DEL "CHAIN OF THOUGHT" (Catena di pensieri) ---
    # Obblighiamo l'LLM a spiegare il suo ragionamento PRIMA di dare la classificazione.
    # Scrivere i passaggi logici lo costringe a "riflettere", riducendo le
    # allucinazioni e migliorando drasticamente la precisione della scelta.
    # Dico all'LLM di pensare ad alta voce
    ragionamento: str = Field(
        description="Analizza il topic. Spiega se ha una base scientifica, se è troppo generico o se è completamente fuori tema."
    )

    # --- IL BINARIO RIGIDO PER IL ROUTER (Output Strutturato) ---
    # Usiamo 'Literal' per impedire all'LLM di inventarsi risposte o aggiungere chiacchiere.
    # Il modello è COSTRETTO a restituire ESATTAMENTE una di queste 3 parole chiave.
    # Così i nostri 'if' nel grafo (es. if classification == "respond":) non si rompono mai.
    # Do all'LLM il menu a tendina e gli spiego quando usare ogni opzione
    classification: Literal["accept", "refine", "reject", "admin"] = Field(
        description="Scegli 'accept' se l'argomento è minimamente focalizzato. Non essere troppo severo, accetta anche argomenti di media specificità. "
                    "Scegli 'refine' se è scientifico ma troppo vago e necessita di focus. "
                    "Scegli 'reject' se non ha nulla a che fare con la scienza."
                    "Scegli 'admin' se l'utente fa domande sull'agenda, sulle date disponibili o sull'organizzazione, senza chiedere di scrivere un articolo. "
    )
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
#llm_router = llm.with_structured_output(RouterSchema) SBLOCCA QUANDO PASSI A OPEN IA

# 1. Inizializza il parser con il tuo schema
parser = PydanticOutputParser(pydantic_object=RouterSchema)

# 2. Recupera le istruzioni che spiegano al modello come formattare il JSON
format_instructions = parser.get_format_instructions()

# 3. Crea il prompt.
# NOTA: Devi includere {format_instructions} nel prompt di sistema
# e passare il contesto (es. l'email o la query) dell'utente.
router_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un router intelligente. Il tuo compito è analizzare la richiesta e decidere la prossima mossa.\n\n"
        "CONTESTO E REGOLE:\n"
        "{istruzioni_personalizzate}\n\n"
        "REGOLE DI FORMATTAZIONE OBBLIGATORIE:\n"
        "{format_instructions}\n\n"
        "ATTENZIONE: DEVI rispondere ESCLUSIVAMENTE con un oggetto JSON valido che contenga sia 'ragionamento' che 'classification'. "
        "NON aggiungere markdown (come ```json), NON aggiungere spiegazioni extra e NON usare lingue diverse dall'italiano."
    )),
    ("human", "{input}")
])


def clean_llm_json_output(message) -> str:
    # L'LLM restituisce un AIMessage, estraiamo il testo
    text = message.content if isinstance(message, AIMessage) else str(message)

    # Rimuoviamo l'escape non valido sostituendo \' con un semplice '
    cleaned_text = text.replace("\\'", "'")

    # Se noti altri errori comuni (es. sfumature di formattazione), puoi aggiungerli qui
    return cleaned_text


# 2. Aggiorna la chain inserendo la funzione RunnableLambda in mezzo
llm_router = router_prompt | llm | RunnableLambda(clean_llm_json_output) | parser
def triage_prompt(state: State) -> Command[Literal["response_agent", "__end__"]]:

    system_prompt = triage_system_prompt.format(
        background="Ciao compare, ",
        triage_instructions=""
    )

    user_prompt = state["prompt_input"]

    result = llm_router.invoke(
        {
            "istruzioni_personalizzate": system_prompt,
            "format_instructions": format_instructions,
            "input": user_prompt
        }
    )

    classification = result.classification

    if classification == "accept":
        print("📧 Classification: ACCEPT - An article can be written about this topic")
        goto = "response_agent"

        istruzioni_operative = (
            "Sei un assistente editoriale. Il tuo compito è scrivere articoli e aiutare a pubblicarli.\n"
            "REGOLA D'ORO: Fai UNA cosa alla volta e dialoga con l'utente.\n"
            "1. DEVI OBBLIGATORIAMENTE USARE IL TOOL 'write_an_article' per proporre l'articolo. INVENTA TU il contenuto (content), l'autore (author) e il destinatario (to). NON SCRIVERE MAI L'ARTICOLO COME SEMPLICE TESTO NELLA CHAT. Se non chiami il tool, il sistema andrà in crash.\n"
            "2. Attendi che l'utente approvi il testo (ci sarà una pausa umana).\n"
            "3. SOLO DOPO L'APPROVAZIONE, chiedi all'utente quando desidera pubblicarlo o se vuole conoscere le date libere.\n"
            "4. Usa i tool sulle date per aiutarlo a scegliere e infine schedula l'articolo."
        )

        # Add the email to the messages
        update = {
            "classification_decision": result.classification,
            "messages": [{
                    "role": "system",
                    "content": istruzioni_operative
                },
                {"role": "user",
                          "content": f"Write an article: {user_prompt}"
                          }],
        }
    elif result.classification == "reject":
        print("🚫 Classification: REJECT - This topic is off-topic")
        update = {
            "classification_decision": result.classification,
        }
        goto = END
    elif result.classification == "refine":
        # If real life, this would do something else
        print("🔔 Classification: REFINE - This topic is too generic, it needs to be refined")
        goto = "response_agent"  # Invece di END, lo mandiamo all'agente

        # Istruiamo l'agente a chiederti chiarimenti invece di scrivere l'articolo
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user",
                          "content": f"L'argomento '{user_prompt}' è troppo generico. Fai una breve proposta all'utente su come potrebbe restringerlo e chiedigli cosa preferisce."
                          }],
        }
    elif classification == "admin":
        print("📅 Classification: ADMIN - L'utente vuole informazioni sull'agenda")
        goto = "response_agent"

        istruzioni_admin = (
            "Sei un assistente editoriale. L'utente ha fatto una domanda sull'agenda o sulle date. "
            "Usa il tool 'check_daily_number_article' per verificare le disponibilità e rispondi alla sua domanda in modo cortese."
        )

        update = {
            "classification_decision": result.classification,
            "messages": [
                {"role": "system", "content": istruzioni_admin},
                {"role": "user", "content": user_prompt}
            ],
        }
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)


# Creiamo il parser e il prompt collegato all'LLM
feedback_parser = PydanticOutputParser(pydantic_object=HumanFeedbackSchema)

feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un analista. Il tuo compito è capire se l'utente sta approvando l'azione proposta o se chiede delle modifiche.\n\n{format_instructions}"),
    ("human", "Feedback utente: {feedback}")
])

# Questa è la chain che "pensa"
llm_feedback_analyzer = feedback_prompt | llm | feedback_parser


def human_review_node(state: MessagesState):
    """Nodo che mette in pausa il grafo per l'approvazione umana."""
    last_message = state["messages"][-1]

    # 1. Raccogliamo quello che l'LLM vuole fare
    tools_richiesti = last_message.tool_calls
    testo_proposto = last_message.content

    # 2. Creiamo il pacchetto di informazioni
    dati_da_controllare = {
        "testo": testo_proposto,
        "tools": tools_richiesti
    }

    # 3. INTERRUPT: Qui il codice si FREEZA e aspetta la stringa utente
    risposta_utente = interrupt(dati_da_controllare)

    # 4. CHIAMIAMO L'LLM per analizzare la risposta dell'utente!
    print(f"L'utente ha detto: '{risposta_utente}'. Faccio analizzare all'LLM...")

    analisi = llm_feedback_analyzer.invoke({
        "format_instructions": feedback_parser.get_format_instructions(),
        "feedback": risposta_utente
    })

    # 5. Analizziamo l'oggetto Pydantic restituito dall'LLM
    if analisi.approve:
        # Se l'LLM ha capito che approvi, proseguiamo
        print("✅ L'LLM ha confermato l'approvazione.")
        if not tools_richiesti:
            return {"messages": [HumanMessage(content=f"L'utente ha confermato/scelto: {risposta_utente}")]}
        return {"messages": []}
    else:
        # Se l'LLM ha capito che vuoi modifiche, le mandiamo indietro
        print(f"🔄 L'LLM ha rilevato modifiche: {analisi.modifiche}")
        feedback = HumanMessage(
            content=(
                f"L'utente ha rifiutato la proposta o ha richiesto queste modifiche: {analisi.modifiche}. "
                "CRITICAL INSTRUCTION: Analizza il contesto della conversazione e usa il tool corretto "
                "(es. 'write_an_article' se devi riscrivere il testo, 'schedule_date_for_article' se devi cambiare data) "
                "per soddisfare la richiesta. NON rispondere con testo semplice se l'azione richiede un tool."
            )
        )
        return {"messages": [feedback]}


def after_human(state: MessagesState):
    """Decide dove andare dopo il controllo umano."""
    last_message = state["messages"][-1]

    # Se l'ultimo messaggio è un "HumanMessage", vuol dire che hai chiesto modifiche!
    if isinstance(last_message, HumanMessage):
        return "call_llm"  # Torna all'LLM per far correggere il tiro

    # Altrimenti (hai approvato), procedi a chiamare il tool
    return "call_tool"


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

