from langchain.tools import tool

@tool
def write_description(about: str, to: str, content: str):
    """Write a description of microcontroller  """
    return f"My description of microcontroller: {about} to {to} is {content}"