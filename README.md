# Berlin City Services Agent

**Description**  
This repository contains a project that provides an AI assistant for Berlin city services. Users can ask questions about how to obtain documents, register addresses, or other official procedures in Berlin. The assistant retrieves data from the [service.berlin.de](https://service.berlin.de) website and provides step-by-step instructions, along with relevant links.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Gradio Web App](#gradio-web-app)
- [Debugging and Searching](#debugging-and-searching)
- [Scrapy Spiders and Data Embedding](#scrapy-spiders-and-data-embedding)
- [Environment Variables](#environment-variables)
- [Notes on Agent Tools](#notes-on-agent-tools)
- [License](#license)

---

## Overview

This project uses:
- **Scrapy** to crawl and scrape the Berlin service pages.
- **OpenAI embeddings** to vectorize the scraped text.
- **Pinecone** as a vector store backend.
- **LangChain** for building an LLM-based agent with multiple tools (retrieval, summarization, etc.).
- **Gradio** to provide a web-based chat interface.

---

## Features

1. **Scrapy Spider** (`berlin_services_spider.py`):
   - Scrapes Berlin services pages and extracts text/content.

2. **Data Embedding** (`embed_data.py`):
   - Splits scraped text into chunks, encodes them with OpenAI embeddings, and stores them in Pinecone.

3. **Common Utilities** (`common.py`):
   - Contains all initialization code for Pinecone, OpenAI embeddings, vector store, and LLM.

4. **Agent** (`agent.py`):
   - Defines a LangChain-based agent with various tools for retrieval and summarization.

5. **Gradio App** (`app.py`):
   - Runs a local web server where users can chat with the Berlin City Agent.

6. **Search Tool** (`search_tool.py`):
   - For debugging or direct testing of the retrieval chain.

---

## Installation

1. **Clone this repository**  

   git clone https://github.com/YOUR_USERNAME/berlin-city-services-agent.git
   cd berlin-city-services-agent

2.	**Create and activate a virtual environment**

   python -m venv venv

   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate # Windows

3.	**Install dependencies**

   pip install --upgrade pip
   pip install -r requirements.txt

4.	**Set up environment variables**

Create a .env file with your API keys

## Usage

1.	**Scrape the Data**

   scrapy crawl berlin_services -o berlin_services/output.json

This writes scraped data to output.json.

2.	**Embed the Data**

   python embed_data.py

Splits the scraped text, encodes it with OpenAI embeddings, and stores vectors in Pinecone.

3.	**Run the Gradio App**

   python app.py

Opens a local Gradio interface (http://127.0.0.1:7860 by default). You can interact with the chatbot that answers questions about Berlin services.

4.	**(Optional) Run the Search Tool**

   python search_tool.py

Debug or explore direct retrieval QA with sources.


## Contact

For questions or issues, please open an issue on this repo or contact the author.