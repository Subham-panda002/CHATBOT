# CHATBOT
Overview
--------
This project is an AI-powered chatbot built using Python. It can understand user queries, generate intelligent responses, perform sentiment analysis, and manage conversation context. This chatbot can be integrated into websites, applications, or used directly through a terminal.



ENHANCED TIER 1:
----------------
Intelligent Chatbot with Gemini AI (AUTO MODEL DETECTION)
FIXED: Automatically detects and uses available models
No more "model not found" errors!

Features:
- Automatically finds working model
- Falls back to available models if primary fails
- Maintains full conversation history
- Performs sentiment analysis
- Saves conversations to file

Installation:
pip install google-generativeai textblob nltk python-dotenv

Setup:
1. Get Gemini API key from: https://makersuite.google.com/app/apikey
2. Create .env file with: GOOGLE_API_KEY=your_key_here
3. Run: python chatbot_enhanced.py

ENHANCED TIER 2:
----------------
Advanced Intelligent Chatbot with Real-Time Sentiment Analysis
FEATURES:
- Real-time per-message sentiment analysis
- Mood trend tracking across conversation
- Detailed statistics & analytics
- Message-by-message breakdown
- Auto model detection (no errors)
- Full conversation memory
- Advanced analytics dashboard
- Conversation export (JSON/CSV)

Installation:
pip install google-generativeai textblob nltk python-dotenv

Setup:
1. Get Gemini API key from: https://makersuite.google.com/app/apikey
2. Create .env file with: GOOGLE_API_KEY=your_key_here
3. Run: python tier2_chatbot.py

Your characteristics:
- Be conversational and natural
- Keep responses concise but helpful
- Show empathy when appropriate
- Ask clarifying questions if needed
- Be honest if you don't know something
- Maintain context from previous messages
- Respond like a human would

Guidelines:
- Don't make up facts you don't know
- Be respectful and non-judgmental
- Help the user solve their problems
- Engage in meaningful conversation"""
