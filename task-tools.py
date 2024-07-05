import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
import openai

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar a chave da API da OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')


research_agent = Agent(
  role='Pesquisador',
  goal='Encontrar e resumir as últimas notícias sobre IA',
  backstory="""Você é um pesquisador em uma grande empresa.
  Você é responsável por analisar dados e fornecer insights
  para o negócio.""",
  verbose=True
)

search_tool = SerperDevTool()

task = Task(
  description='Encontrar e resumir as últimas notícias sobre IA',
  expected_output='Um resumo em forma de lista dos 5 principais eventos mais importantes de IA',
  agent=research_agent,
  tools=[search_tool]
)

crew = Crew(
    agents=[research_agent],
    tasks=[task],
    verbose=2
)

result = crew.kickoff()
print(result)
