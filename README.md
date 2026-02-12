# Ploty Dash App with Conversational Recommender via LangChain

The provided code defines a Dash web application aimed at demonstrating the integration of advanced natural language processing (NLP) techniques for semantic search and interactive chat functionalities

## Key Features:
- ***Semantic Search:*** Leverages embeddings to enhance search capabilities beyond simple keyword matches.
- ***Interactive Chat:*** Utilizes a conversational AI to engage users, providing multilingual responses and relevant information sourced from a database.
- ***User-Friendly Interface:*** Easy navigation through various features such as home, list view, chat, and background information sections.

## Technologies Used:
- ***Dash:*** For building the interactive web application.
- ***OpenAI:*** For embeddings and conversational AI models.
- ***LangChain:*** For managing the conversational AI operations.
- ***FAISS:*** For efficient similarity search in high dimensional spaces.
- ***Pandas and NumPy:*** For data manipulation and operations.
- ***TikToken:*** For tokenizing text data to handle input lengths effectively.

## Application Structure:
- ***Initialization:*** Sets up the Dash app, API keys, and styling.
- ***Data Loading and Preprocessing:*** Involves loading project data and transforming it for use with OpenAI's embedding models.
- ***Embedding and Search Functions:*** Functions to compute embeddings and perform cosine similarity searches.
- ***Chat System Initialization:*** Integrates various components like embeddings, vector storage, and conversational AI to handle user interactions.
- ***Web Page Layouts:*** Defines the layout for different sections of the app, each designed to handle specific functionalities like search and chat.
- ***Callbacks:*** Manage interactions within the application, such as updating search results, handling chat messages, and navigating between pages.
