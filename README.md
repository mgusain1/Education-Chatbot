The University Chatbot is an AI-powered assistant designed to help students, parents, and staff quickly access important information about universities. It uses natural language processing (NLP) and a vector database to provide instant answers about admission requirements, tuition, test scores, financial aid, and more.

Instead of searching through dozens of web pages, users can simply ask questions in plain English (e.g., “What’s the average SAT score for Stanford?” or “How much is tuition at NYU for international students?”) and receive accurate, data-backed responses.

This project is built to demonstrate how large language models (LLMs) can be combined with structured datasets (like the College Scorecard) to create a scalable and reliable university Q&A assistant.

🚀 Features

Natural Language Search – Ask questions like you would in conversation.

University Metadata Integration – Includes tuition, SAT/ACT averages, location, and more.

Semantic Search with Vector Database – Uses embeddings + FAISS to retrieve relevant university data.

FastAPI Backend – Exposes clean APIs for querying the chatbot.

Extensible Design – Can be extended to handle events, courses, or even scheduling with university APIs.

🛠️ Tech Stack

Backend: FastAPI

LLM: OpenAI (text-embedding-ada-002 for embeddings, GPT for Q&A)

Database: FAISS (vector store) + CSV metadata

Data Source: College Scorecard (merged with custom university dataset)

Frontend (optional): Streamlit (for testing chatbot UI)
