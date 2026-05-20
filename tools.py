from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel

DB_ARTICOLI_PUBBLICATI = {
    "2026-05-19": ["Novità su ROS 3", "Boston Dynamics Update", "Review CyberDog 3"], # Oggi ci sono 3 articoli
    "2026-05-20": ["I droni del futuro", "AI e Robotica industriale"]                  # Domani ce ne sono 2
}

@tool
def write_an_article(about: str, to: str, content: str):
    """Write an article about the topic  """
    return f"My article about: {about} for {to} is {content}"


@tool
def check_daily_number_article(date: str)->int:
    """Check the daily number article"""
    daily_articles = DB_ARTICOLI_PUBBLICATI.get(date, [])
    return len(daily_articles)

@tool
def schedule_date_for_article(title: str, author:str, date:str)->str:
    """Schedule date for article
    Use date format 'YYYY-MM-DD'"""
    DB_ARTICOLI_PUBBLICATI[date].append(title)
    return ""

@tool
def triage_email(category: Literal["ignore", "notify", "write_article"]) -> str:
    """Triage an email into one of three categories: ignore, notify, respond."""
    return f"Classification Decision: {category}"

@tool
class Done(BaseModel):
    "E-mail sent successfully"
    done: bool