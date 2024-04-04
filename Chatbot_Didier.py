import gradio as gr
import os
import pandas as pd
from pprint import pprint

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain 

# from langchain.agents import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import Tool, load_tools, initialize_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

def make_prompt():
    
    system_query_template = """
    Tu es Didier, un Chatbot spécialisé dans les réponses aux questions sur le football. 
    Tu dois aider les équipes et les recruteurs à trouver les meilleurs joueurs pour leur équipe. Tu es donc un fin spécialiste du football.

    Tu réponds toujours dans la même langue que la question qui est posée.
    Si on te pose des questions qui n'ont pas de lien avec le football, tu dois répondre que tu n'es pas capable de répondre à la question.
    Dans tes réponses, tu dois directement répondre aux questions, ne dis pas "Bonjour" ou d'autres formules de politesse.

    
    ---------------------- Contexte dont tu dois te baser pour répondre à la question :
    {context}
    ----------------------

    ---------------------- Historique de la conversation :
    {history}
    ----------------------
    """

    user_query_template = """
    ---------------------- Question : 
    {input_text}
    ----------------------
    """

    message = [
        SystemMessagePromptTemplate.from_template(system_query_template),
        HumanMessagePromptTemplate.from_template(user_query_template)
    ]
    prompt = ChatPromptTemplate.from_messages(message)
    return prompt

def load_pdf(pdf_url):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True)
    loader_pdf = PyPDFLoader(pdf_url)
    documents_pdf = loader_pdf.load()
    documents_pdf = text_splitter.split_documents(documents_pdf)
    return documents_pdf

def load_csv(csv_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True)
    loader_csv = CSVLoader(file_path=csv_path, encoding='utf-8', csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['Joueur', 'Poste', 'Pays', 'Indicateur', 'RIG', 'DIS', 'PER', 'REC', 'DAN', 'GAR']
    })
    documents_csv = loader_csv.load()
    documents_csv = text_splitter.split_documents(documents_csv)
    return documents_csv


def chatbot(input, history):

    ########## Initialisation du modèle de langage et des autres fonctionnalités

    llm = Ollama(model="llama2", temperature=0.5) # dolphin-phi
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    output_parser = StrOutputParser()

    # ???????? Utilité/utilisation de ConversationBufferMemory ?

    ########## Base de données vectorielle

    embeddings = OllamaEmbeddings()
    documents = []
    
    # Avec un fichier PDF ...
    pdf_url = "https://ekladata.com/HKXjuvmix8Ka-6RULZEarVlAWjY/regles-foot.pdf"
    documents_pdf = load_pdf(pdf_url)
    documents.extend(documents_pdf)

    # ... et un fichier CSV
    csv_path = "documents/test_db.csv"
    documents_csv = load_csv(csv_path)
    documents.extend(documents_csv)

    db = Chroma.from_documents(documents, embeddings, persist_directory='./documents_vect')

    # ???????? text_splitter spécifique pour les fichiers CSV ?
    # ???????? Tester aussi en mettant tout dans documents et en splittant ensuite

    ########## Retriever
    retriever = db.as_retriever(search_type="similarity", search_wargs={"k": 3})
    similar_docs = retriever.get_relevant_documents(input)
    print("######## Documents similaires :")
    for document in similar_docs:
        print("###")
        pprint(document)
    print("\n")

    ########## Chaîne de traitement

    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     chain_type="stuff", # Voir à quoi ça correspond
    #     retriever=retriever,
    #     return_source_documents=True,
    #     combine_docs_chain_kwargs={'prompt': make_prompt()}
    # )
    # reponse = chain({"input_text" : input,
    #                  "context" : similar_docs,
    #                  "history" : history})

    prompt = make_prompt()

    chain = prompt | llm | output_parser
    print("######## Chaine :")
    print(chain)
    print("\n")

    reponse = chain.invoke({"input_text" : input,
                            "context" : similar_docs,
                            "history" : history})
    
    return reponse


chatbot_v1 = gr.ChatInterface(
    fn=chatbot, 
    # inputs="text", 
    # outputs="text",
    title="Didier le Chatbot - V1.0",
    description="Tappez votre message et appuyez sur Entrée pour obtenir une réponse de la part de Didier sur le football.")

chatbot_v1.launch(share=True)


""" A faire :

################### re-implémentation d'un csv agent

path_to_csv = "data/test_db.csv" # on peut mettre une liste pour en avoir plusieurs
agent = create_csv_agent(
    llm,
    path_to_csv,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

################### changer la similarité (cf msg Yves)
pas possible avec Chroma ?

################### Si on ne fournit que le csv, le retriever y pioche de l'info mais si il y a aussi le pdf le retriever ne ramène plus rien du csv

"""





""" A explorer :

https://nanonets.com/blog/langchain/
combining BM25 and FAISS

Notamment pour la partie ConversationalRetrievalChain :
https://github.com/ksm26/LangChain-Chat-with-Your-Data/blob/main/L4-Retrieval.ipynb


Tuto complet :
https://realpython.com/build-llm-rag-chatbot-with-langchain/



https://github.com/AhmedEwis/AI_Assistant

https://betterprogramming.pub/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339

"""



# from langchain_community.document_loaders import Docx2txtLoader
# from langchain_community.document_loaders import TextLoader
# documents = []
# for file in os.listdir('docs'):
#     if file.endswith('.pdf'):
#         pdf_path = './docs/' + file
#         loader = PyPDFLoader(pdf_path)
#         documents.extend(loader.load())
#     elif file.endswith('.docx') or file.endswith('.doc'):
#         doc_path = './docs/' + file
#         loader = Docx2txtLoader(doc_path)
#         documents.extend(loader.load())
#     elif file.endswith('.txt'):
#         text_path = './docs/' + file
#         loader = TextLoader(text_path)
#         documents.extend(loader.load())



    # df = pd.read_csv("data/test_db.csv")
    # agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # chain = prompt | agent | output_parser







"""
tools = load_tools()
llm_general_tool = Tool(name="Language Model", func=chain.run, description="Use this tool for general purpose queries and logic.")
tools.append(llm_general_tool)
llm_pandas_tool = create_pandas_dataframe_agent(llm, pd.read_csv("data/test_db.csv"), verbose=True)

general_agent = initialize_agent(agent="zero-shot-react-description", tools=tools, llm=llm, verbose=True)

"""