from typing import Literal, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class StateInput(TypedDict):
    prompt_input: str
    
class State(MessagesState):
    # This state class has the messages key build in
    prompt_input: str
    classification_decision: Literal["reject", "refine", "accept", "admin"]

class HumanFeedbackSchema(BaseModel):
    approve: bool = Field(description="Imposta a True se l'utente approva. Imposta a False se chiede modifiche.")
    modifiche: str = Field(description="Se approve è False, riassumi le modifiche richieste. Se True, lascia vuoto.")