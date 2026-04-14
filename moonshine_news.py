from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from openai import OpenAI
import os

API_KEY = os.environ["OPEN_AI_KEY"] # non standard key name

client = OpenAI(api_key=API_KEY)

def get_news(topic):
    try:
        raw_news = client.responses.create(
            model = "gpt-5.4-nano",
            tools = [{"type": "web_search"}],
            input = f"""Wyszukaj w sieci najciekawszą, pozytywną wiadomość na temat {topic}.
            Zwróć krótki rzeczowy opis faktów"""
        )
        return raw_news.output_text
    except Exception as e:
        return f"Coś poszło nie tak, niusów nie będzie: {e}"

class InfoModel(BaseModel):
    title: str = Field(description="Tytuł wieści w stylu Wędrowycza")
    body: str = Field(description="Treść wieści, zachowująca jako tako sens aczkolwiek barwna i doprawiona oparami piwa")

llm = init_chat_model(
    api_key = API_KEY,
    model = "openai:gpt-5.4-nano"
)

structured_llm = llm.with_structured_output(
    InfoModel,
)

with open("wedrowycz_lore.txt", "r", encoding="utf-8") as f:
    personality = f.read()

### all mistakes in spelling are on purpose to help LLM feel vibe od Wedrowycz
jakub_prompt = """Jesteś Jakub Wędrowycz, historia którą trochę pamiętassz a troche nie to: 
    {personality}. Opowiedz znajomemu przy piwie własnymi słowami historię którą ostatnio usłyszałeś. Wczuj się w role odpowiadaj jak Wędrowycz.
    Wczuj się w role, to jest chłop z lubelszczyzny proste wypowiedzi"""

prompt = ChatPromptTemplate([
    ("system", jakub_prompt),
    ("user", "Treść wiadomości do streszczenia i opowiedzenia: {raw_news}")
])

question = "Działo się coś w Warszawie z piwem ostatnio?"

raw_news = get_news(question)
print(raw_news) # for debugging

chain = prompt | structured_llm

answer = chain.invoke({"raw_news": raw_news, "personality": personality})

print(f"Tytuł newsa: {answer.title}")
print(f"Wieści: {answer.body}")