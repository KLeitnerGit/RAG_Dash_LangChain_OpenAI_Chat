#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[3]:


# Libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
from getpass import getpass
from fast_dash import FastDash, Fastify, dcc, dmc
import re
import tiktoken
import warnings

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

warnings.filterwarnings("ignore")


# Initialisation of the Dash app 
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "R&D Semantic Search & Chat (proof of concept)"

# Setting Colours
colors = {
    'background': '#E7E8D1',
    'navbar': '#A7BEAE',
    'text': '#B85042'
}

typography = {
    'font_family': 'Arial, sans-serif'
}

# Navigation layout
nav_link_style = {
    'color': colors['text'],
    'padding': '30px 0',
    'display': 'block',
    'textDecoration': 'none',
    'fontSize': '20px',
    'fontWeight': 'bold',
    'margin': '15px 0'
}

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")
openai_api_key_component = dmc.PasswordInput(
    id='api-key-input',
    placeholder="API Key",
    description="Get yours at https://platform.openai.com/account/api-keys",
    required=True,
)


@app.callback(
    Output('api-key-output', 'children'),
    [Input('api-key-input', 'value')]
)
def update_openai_api_key(api_key_value):
    if api_key_value:
        openai.api_key = api_key_value
        return "API Key Updated!"  
    return "Please enter the API Key."


chat_history = []

# Load  data
df7 = pd.read_csv("text_embeeding_ADA02_MIX.csv")
df = df7.drop("Unnamed: 0", axis=1)
df.ada_v2_embedding_EN = df.ada_v2_embedding_EN.apply(ast.literal_eval).apply(np.array)

# Define functions for embeddings and cosine sim
def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def search_docs(df, user_query, top_n=40, model="text-embedding-ada-002"):
    embedding = get_embedding(user_query, model=model)
    embedding = np.array(embedding).reshape(1, -1)
    df["similarities"] = df.ada_v2_embedding_EN.apply(lambda x: cosine_similarity(x.reshape(1, -1), embedding)[0][0])
    return df.sort_values("similarities", ascending=False).head(top_n).reset_index(drop=True)

# Initialize chat 
def initialize_chat_system():
    # Load data and get length token
    loader = CSVLoader(file_path='projects_small.csv', source_column="url_new", encoding="utf-8")
    data_projects1 = loader.load()  # Encode
    tokenizer = tiktoken.get_encoding('cl100k_base') 

    def tiktoken_len(text):
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)  
    
    # Recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""])  # Split documents

    docs_project1 = text_splitter.split_documents(data_projects1)

    # Set embedding model
    model_name = 'text-embedding-ada-002'
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai.api_key)

    # Set vectorstore
    dl_FAI_projects = FAISS.from_documents(docs_project1, embeddings)
    dl_FAI_projects.save_local("dl_FAI_projects")
    db_projects = FAISS.load_local("dl_FAI_projects", embeddings)  
    
    # Set chat model
    llm = OpenAI(openai_api_key=openai.api_key, temperature=0,
        model_name="gpt-3.5-turbo-16k",
        max_tokens=700)  
    
    # Setting prompt
    template = (
    "You are a friendly Q&A bot. A highly intelligent system that answers user {question} based only on the information provided by the user above each question. "
    "If the information cannot be found in the information provided by the user, you truthfully say 'I don't know'. "
    "You have multilingual capabilites and always detect the language of the user {question} and provide answer in the same language as the {question}, except the project titel"
    "Take note of the sources and include them in the answer in the format: 'SOURCES: source1 source2', use 'SOURCES' in capital letters regardless of the number of sources. "
    "Combine the chat history and follow up question into a standalone question. Chat History: {chat_history} Follow up question: {question}"
)

    # Setting conversational retrieval chain
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(template)  

    question_generator_chain = LLMChain(llm=llm, prompt=CUSTOM_QUESTION_PROMPT)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=10)
    qa = ConversationalRetrievalChain(
        retriever=db_projects.as_retriever(k=5),
        question_generator=question_generator_chain,
        combine_docs_chain=doc_chain, memory=memory)

    return qa

qa = initialize_chat_system()


# Definition of app layout 
app.layout = html.Div(style={'backgroundColor': colors['background'], 'fontFamily': typography['font_family'], 'display': 'flex', 'height': '100vh'}, children=[
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('Home', href='/', style=nav_link_style),
        html.Div(openai_api_key_component, style={'marginTop': '20px'}),  # Add the API key input component to the navbar
        dcc.Link('List-View', href='/list-view', style=nav_link_style),
        dcc.Link('Chat', href='/chat', style=nav_link_style),
        dcc.Link('Background', href='/background', style=nav_link_style),
    ], style={'width': '300px', 'backgroundColor': colors['navbar'], 'padding': '30px 0', 'textAlign': 'center'}),
    html.Div(id='page-content', style={'flex': 1, 'padding': '40px', 'overflowY': 'auto'})
])

# Placeholder content for the home page
home_layout = html.Div([
    html.H1("R&D Semantic Search & Chat (proof of concept)", style={'color': colors['text'], 'textAlign': 'center', 'fontSize': '36px'}),
    
    html.Div([
        html.H3('Introduction', style={'color': colors['text'], 'fontSize': '28px'}),
        html.P([
        "Welcome to the R&D Semantic Search & Chat (proof of concept) Project. This tool was developed as a proof of concept for my master's thesis. The goal is to harness the capabilities of cutting-edge machine learning models to enhance semantic search and interactive chat functionalities. This serves as an augmentation to the existing solution found at ",
        html.A("HSLU's project page", href='https://www.hslu.ch/en/lucerne-university-of-applied-sciences-and-arts/research/projects/', target="_blank"),
        ". Dive in, explore the comprehensive knowledge base of HSLU projects, and find answers to your questions."
    ], style={'fontSize': '18px'}),
        html.H3('Features', style={'color': colors['text'], 'fontSize': '28px'}),
        html.Ul([
            html.Li('Semantic Search: Find relevant documents based on the meaning of your query, not just keyword matches.', style={'fontSize': '18px'}),
            html.Li('Interactive Chat: Engage with the chatbot to get multilingual answers and insights and sources.', style={'fontSize': '18px'}),
            html.Li('User-friendly Interface: Easily navigate and get the information you need.', style={'fontSize': '18px'}),
        ]),
        
        html.H3('How to Use', style={'color': colors['text'], 'fontSize': '28px'}),
        html.P('1. Start by entering your OpenAI API key.', style={'fontSize': '18px'}),
        html.P('2. Navigate to the List-View to perform semantic searches.', style={'fontSize': '18px'}),
        html.P('3. Use the Chat section to interact with the AI chatbot.', style={'fontSize': '18px'}),
        
        html.H3('Contact', style={'color': colors['text'], 'fontSize': '28px'}),
        html.P('For questions, feedback, or support, please reach out to kathrin.leitner@protonmail.com', style={'fontSize': '18px'}),
    ], style={'margin': '20px 10%'}),
])


# Background layout
background_layout = html.Div([
    html.H1('Background Information', style={'color': colors['text'], 'textAlign': 'center', 'fontSize': '36px'}),
    
    html.Div([
        html.H3('Problem:', style={'color': colors['text'], 'fontSize': '28px'}),
        html.P("Knowledge management, especially identifying expertise, is crucial for organisational success. However, it remains a significant challenge for many organisations. A recent employee survey at the Lucerne University of Applied Sciences and Arts (HSLU) identified knowledge management and digitalisation as areas needing improvement."),
        
        html.H3('Aim:', style={'color': colors['text'], 'fontSize': '28px'}),
        html.P("This pilot study aims to determine whether word embedding techniques using transformer-based mono- and multilingual pre-trained language models - namely 'paraphrase-MiniLM-L6-v2' and 'paraphrase-multilingual-MiniLM-L12-v2' from the Siamese Bert Network (SBERT), as well as OpenAI's 'text-embedding-ada-002' and Cohere's 'multilingual-22-12' model - can produce more relevant search results in a meaningful ranking than the conventional Tf-IDF method currently used within HSLU's expertise search framework."),
        
        html.H3('Method:', style={'color': colors['text'], 'fontSize': '28px'}),
        html.P("To address the primary research question, comparative research is conducted by implementing knowledge-based recommender systems within a multilingual framework. Specifically, the study compares the results of ten information needs (queries) in three different languages (German, English, Spanish) across HSLU projects in three language versions (German only, English only, and a mix of German and English)."),
        
        html.H3('Findings:', style={'color': colors['text'], 'fontSize': '28px'}),
        html.P("The evaluation reveals that the multilingual and closed-source embedding models of OpenAI and Cohere outperform the open-source models of SBERT and TF-IDF approaches and thus could significantly improve HSLU expertise search. Notably, OpenAI slightly beat Cohere in this study setting."),
        
        html.H3('Solution:', style={'color': colors['text'], 'fontSize': '28px'}),
        html.P("Using the best-performing embedding model from the evaluation, specifically OpenAI's 'text-embedding-ada-002', a retrieval-augmented conversational recommender system was constructed. This system uses the LangChain library and OpenAI's gpt-3.5-turbo-16k as its conversation model. It also integrates a conversation buffer memory that stores the last four interactions."),
        
        html.H3('Limitations:', style={'color': colors['text'], 'fontSize': '28px'}),
        html.Ul([
            html.Li("As a pilot study, this project is anchored on project data extracted as of December 2022."),
            html.Li("The list view provides a comprehensive overview of projects relevant to a specific query. However, the conversational interface limits its display to the top 5 results to maintain optimal performance, even when using the expanded context window size with the gpt-3.5 turbo-16k model."),
            html.Li("Although the primary prompt is configured to recognise the language of the user's query and respond in kind, its performance in this regard could be more consistent."),
            html.Li("As the research only looked at publicly available data, the scope of the knowledge base is limited.")
        ]),
    ], style={'margin': '20px 10%'}),
])

# List-View layout
list_view_layout = html.Div([
    html.H1('List View', style={'color': colors['text'], 'textAlign': 'center', 'fontSize': '36px'}),
    html.Div([
        dcc.Input(id='user-query', type='text', placeholder='Enter search query here...', style={'width': '60%'}),
        html.Button('Search', id='search-button', style={'width': '18%', 'marginLeft': '2%'}),
    ]),
    html.Div(id='search-results', style={'marginTop': '20px'})
])



# Chat layout
chat_layout = html.Div([
    html.H1('Chat', style={'color': colors['text'], 'textAlign': 'center', 'fontSize': '36px'}),
    dcc.Loading(id="loading", type="circle", children=[
        html.Div(id='chat-display', style={'height': '400px', 'overflowY': 'scroll', 'marginBottom': '20px'}),
    ]),
    dcc.Input(id='chat-input', type='text', placeholder='Type your message...', style={'width': '60%'}),
    html.Button('Send', id='send-button', style={'width': '18%', 'marginLeft': '2%'}),
    html.Button('Clear', id='clear-button', style={'width': '18%', 'marginLeft': '2%'}),
    dcc.Interval(id='interval-component', interval=1*1000, max_intervals=0),  # 1 second per interval
    html.Div(id='hidden-div', style={'display':'none'})
])


@app.callback(
    [Output('chat-display', 'children'),
     Output('chat-input', 'value'),
     Output('interval-component', 'max_intervals')],
    [Input('send-button', 'n_clicks'),
     Input('clear-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [dash.dependencies.State('chat-input', 'value')]
)
def update_chat(send_clicks, clear_clicks, n_intervals, user_message):
    global chat_history
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Type a message to start the conversation.", "", 0

    if 'clear-button' in ctx.triggered[0]['prop_id']:
        chat_history = []
        return "Chat cleared. Type a message to start a new conversation.", "", 0

    if not user_message:
        return "Type a message to start the conversation.", "", 0

    
    result = qa({"question": user_message, "chat_history": chat_history})
    response = result['answer']

    # Get href to display  clickable links
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response)
    for url in urls:
        response = response.replace(url, f'[{url}]({url})')

    chat_history.append({"user": user_message, "bot": response})

    # Update the chat display with the user message and the response
    chat_bubbles = []
    for chat in chat_history:
        user_msg = html.Div(f'User: {chat["user"]}', style={'backgroundColor': '#E7E8D1', 'padding': '10px', 'borderRadius': '15px 15px 15px 0'})
        bot_msg = html.Div(dcc.Markdown(chat["bot"]), style={'backgroundColor': '#A7BEAE', 'padding': '10px', 'borderRadius': '15px 15px 0 15px', 'marginTop': '10px'})
        chat_bubbles.append(html.Div([user_msg, bot_msg], style={'marginBottom': '20px'}))

    return chat_bubbles, "", 0


@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [dash.dependencies.State('user-query', 'value')]
)
def update_output(n_clicks, value):
    if not value:
        return "Please enter a query to search."
    
    results = search_docs(df, value)
    children = []
    for idx, row in results[["Projekttitel", "Abstract", "url_new", "similarities"]].iterrows():
        children.append(html.Div([
            html.H5(f"Result {idx+1}:"),
            html.P(f"Projekttitel: {row['Projekttitel']}"),
            html.P(f"Abstract: {row['Abstract']}"),
            html.A(f"URL: {row['url_new']}", href=row['url_new']),
            html.P(f"Similarity Score: {row['similarities']:.4f}")
        ], style={'border': '1px solid', 'padding': '10px', 'margin': '10px 0'}))
    return children

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/background':
        return background_layout
    elif pathname == '/list-view':
        return list_view_layout
    elif pathname == '/chat':
        return chat_layout
    else:
        return home_layout


if __name__ == '__main__':
    app.run_server(debug=True, port=8192)

