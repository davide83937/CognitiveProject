from typing import Literal
import calendar

from prompts import triage_system_prompt
from langchain.tools import tool
from pydantic import BaseModel

DB_ARTICOLI_PUBBLICATI = {
    "2026-05-19": ["Novità su ROS 3", "Boston Dynamics Update", "Review CyberDog 3"], # Oggi ci sono 3 articoli
    "2026-05-20": ["I droni del futuro", "AI e Robotica industriale"]                  # Domani ce ne sono 2
}

@tool
def write_an_article(about: str, to: str, content: str):
    """Write an article about the topic given by user, author is IA, article is for people  """
    return f"My article about: {about} for {to} is {content}"


@tool
def check_daily_number_article(date: str)->int:
    """Check the daily number article"""
    daily_articles = DB_ARTICOLI_PUBBLICATI.get(date, [])
    return len(daily_articles)

@tool
def schedule_date_for_article(title: str, author:str, date:str)->str:
    """Schedule date for article.
    Use date format 'YYYY-MM-DD'.
    IMPORTANT: Before calling this tool, ALWAYS check the number of articles for that date
    using 'check_daily_number_article'. If there are already 3 or more articles, DO NOT schedule
    on that date and look for the next available day."""
    if date is not DB_ARTICOLI_PUBBLICATI:
        DB_ARTICOLI_PUBBLICATI[date]=[]
    DB_ARTICOLI_PUBBLICATI[date].append(title)
    return ""


@tool
def get_available_dates_in_month(year: str, month: str) -> str:
    """Use this tool when the user asks for available dates in a specific month and year.
    Provide year (e.g., '2026') and month (e.g., '05' or '06')."""

    y = int(year)
    m = int(month)

    # calendar.monthrange restituisce il giorno della settimana di inizio e il numero totale di giorni nel mese
    _, num_days = calendar.monthrange(y, m)

    date_libere = []

    # Cicliamo su tutti i giorni del mese richiesto
    for day in range(1, num_days + 1):
        # Formattiamo la data nel formato YYYY-MM-DD per farla combaciare con il DB
        data_str = f"{y:04d}-{m:02d}-{day:02d}"

        # Controlliamo quanti articoli ci sono in quella specifica data (se non esiste, restituisce [] e quindi 0)
        articoli_presenti = len(DB_ARTICOLI_PUBBLICATI.get(data_str, []))

        # Se la data non è piena (meno di 3 articoli), la aggiungiamo alla lista delle date libere
        if articoli_presenti < 3:
            date_libere.append(data_str)

    if not date_libere:
        return f"Mi dispiace, non ci sono date libere nel {month}/{year}."

    # Restituiamo le prime 7 date libere per non intasare la memoria del modello
    date_mostrate = ", ".join(date_libere[:7])

    if len(date_libere) > 7:
        return f"Ci sono molte date libere nel {month}/{year}. Le prime disponibili sono: {date_mostrate}."
    else:
        return f"Le date libere nel {month}/{year} sono: {date_mostrate}."

@tool
class Done(BaseModel):
    "E-mail sent successfully"
    done: bool