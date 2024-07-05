import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)
from langchain_community.chat_models import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar as chaves da API
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Inicializar ferramentas
docs_tool = DirectoryReadTool(directory='./blog-posts')
file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Criar agentes
researcher = Agent(
    role='Analista de Pesquisa de Mercado',
    goal='Fornecer análise de mercado atualizada sobre a indústria de IA',
    backstory='Um analista especialista com um olhar atento para as tendências de mercado.',
    tools=[search_tool, web_rag_tool],
    verbose=True
)

writer = Agent(
    role='Escritor de Conteúdo',
    goal='Criar posts envolventes sobre a indústria de IA',
    backstory='Um escritor habilidoso com paixão por tecnologia.',
    tools=[docs_tool, file_tool],
    verbose=True
)

# Definir tarefas
research = Task(
    description='Pesquisar as últimas tendências na indústria de IA e fornecer um resumo.',
    expected_output='Um resumo dos 3 principais desenvolvimentos em tendência na indústria de IA com uma perspectiva única sobre sua importância.',
    agent=researcher
)

write = Task(
    description='Escrever um post envolvente sobre a indústria de IA, baseado no resumo do analista de pesquisa. Inspirar-se nos últimos posts do diretório.',
    expected_output='Um post de blog de 4 parágrafos formatado em markdown com conteúdo envolvente, informativo e acessível, evitando jargões complexos.',
    agent=writer,
    output_file='blog-posts/novo_post.md'  # O post final será salvo aqui
)

# Montar a equipe
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=2,
    manager_llm= llm
)

# Executar as tarefas
crew.kickoff()