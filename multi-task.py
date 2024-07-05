import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_community.chat_models import ChatOpenAI
import openai

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar a chave da API da OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Inicializar o modelo de linguagem natural se necessário
# llm = ChatOpenAI(model='gpt-3.5-turbo')

# Definir o agente Pesquisador
research_agent = Agent(
    role='Pesquisador',
    goal='Encontrar e resumir as últimas notícias sobre IA',
    backstory="""Você é um pesquisador em uma grande empresa.
    Você é responsável por analisar dados e fornecer insights
    para o negócio.""",
    verbose=True,
    # llm=llm
)

# Ferramenta de pesquisa SerperDevTool
search_tool = SerperDevTool()

# Tarefa: Encontrar e resumir as últimas notícias sobre IA
research_ai_task = Task(
    description='Encontrar e resumir as últimas notícias sobre IA',
    expected_output='Um resumo em forma de lista dos 5 principais eventos mais importantes de IA',
    async_execution=True,
    agent=research_agent,
    tools=[search_tool]
)

# Tarefa: Encontrar e resumir as últimas notícias sobre AI Ops
research_ops_task = Task(
    description='Encontrar e resumir as últimas notícias sobre AI Ops',
    expected_output='Um resumo em forma de lista dos 5 principais eventos mais importantes de AI Ops',
    async_execution=True,
    agent=research_agent,
    tools=[search_tool]
)

# Definir o agente Escritor
writer_agent = Agent(
    role='Escritor',
    goal='Escrever um post completo sobre a importância da IA e suas últimas notícias',
    backstory="""Você é um escritor em uma grande empresa.
    Sua responsabilidade é criar conteúdo envolvente e informativo
    sobre tecnologias emergentes como a inteligência artificial.""",
    verbose=True,
    # llm=llm
)

# Tarefa: Escrever um post completo sobre IA usando os resultados das tarefas de pesquisa
write_blog_task = Task(
    description="Escrever um post completo sobre a importância da IA e suas últimas notícias",
    expected_output='Um post completo com 2 parágrafos',
    agent=writer_agent,
    context=[research_ai_task, research_ops_task]
)

# Configurar a equipe
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_ai_task, research_ops_task, write_blog_task],
    verbose=2,
    # manager_llm=llm
)

# Iniciar as tarefas da equipe
result = crew.kickoff()
print(result)
