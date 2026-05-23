from typing import Literal, TypedDict

from langgraph.graph import MessagesState

class StateInput(TypedDict):
    prompt_input: str
    
class State(MessagesState):
    # This state class has the messages key build in
    prompt_input: str
    classification_decision: Literal["reject", "refine", "accept"]