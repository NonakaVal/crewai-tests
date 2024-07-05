import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOpenAI
import openai

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar a chave da API da OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define seus agentes
pesquisador = Agent(
    role="Pesquisador",
    goal="Conduzir pesquisa detalhada e análise sobre IA e agentes de IA",
    backstory="Você é um pesquisador especializado em tecnologia, engenharia de software, IA e startups. Trabalha como freelancer e está atualmente pesquisando para um novo cliente.",
    allow_delegation=False,
)

escritor = Agent(
    role="Escritor Sênior",
    goal="Criar conteúdo envolvente sobre IA e agentes de IA",
    backstory="Você é um escritor sênior especializado em tecnologia, engenharia de software, IA e startups. Trabalha como freelancer e está atualmente escrevendo conteúdo para um novo cliente.",
    allow_delegation=False,
)

# Define sua tarefa
tarefa = Task(
    description="Gerar uma lista de 5 ideias interessantes para um artigo. Em seguida, escreva um parágrafo cativante para cada ideia que demonstre o potencial de um artigo completo sobre este tema. Retorne a lista de ideias com seus parágrafos e suas anotações.",
    expected_output="5 tópicos, cada um com um parágrafo e notas acompanhantes.",
)

# Define o agente gerente
gerente = Agent(
    role="Gerente de Projeto",
    goal="Gerenciar eficientemente a equipe e garantir a conclusão de tarefas de alta qualidade",
    backstory="Você é um gerente de projeto experiente, habilidoso em supervisionar projetos complexos e guiar equipes para o sucesso. Seu papel é coordenar os esforços dos membros da equipe, garantindo que cada tarefa seja concluída dentro do prazo e com o mais alto padrão de qualidade.",
    allow_delegation=True,
)

# Modelo de linguagem natural para o gerente
llm = ChatOpenAI(model='gpt-3.5-turbo')

# Configuração da equipe
equipe = Crew(
    agents=[pesquisador, escritor],
    tasks=[tarefa],
    process=Process.hierarchical,
    manager_llm= llm
)

# Iniciar o trabalho da equipe
resultado = equipe.kickoff()
