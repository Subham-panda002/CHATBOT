"""
ENHANCED TIER 1: Intelligent Chatbot with Gemini AI (AUTO MODEL DETECTION)
===========================================================================
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
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from textblob import TextBlob
import json

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')

if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY not found!")
    print("\nTo fix this:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Create an API key")
    print("3. Create a .env file in this folder")
    print("4. Add: GOOGLE_API_KEY=your_key_here")
    print("5. Run this script again")
    exit()

genai.configure(api_key=API_KEY)


# ============================================================================
# PART 1: MODEL DETECTOR (Find Available Models)
# ============================================================================

class ModelDetector:
    """Automatically detects available Gemini models"""
    
    @staticmethod
    def get_available_models():
        """Get list of models that support generateContent"""
        try:
            models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    models.append(model.name.replace('models/', ''))
            return models
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
            return []
    
    @staticmethod
    def get_best_model():
        """Get the best available model (in order of preference)"""
        preferred_models = [
            'gemini-2.0-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro',
            'gemini-pro-vision'
        ]
        
        available = ModelDetector.get_available_models()
        
        # Try preferred models first
        for model in preferred_models:
            if model in available:
                return model
        
        # Use first available model
        if available:
            return available[0]
        
        # Fallback
        return None


# ============================================================================
# PART 2: CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Manages entire conversation history"""
    
    def __init__(self):
        self.user_messages = []
        self.bot_responses = []
        self.chat_history = []
        self.start_time = datetime.now()
        self.conversation_file = "conversation_history.txt"
    
    def add_exchange(self, user_msg, bot_msg):
        """Record a complete exchange"""
        self.user_messages.append(user_msg)
        self.bot_responses.append(bot_msg)
        
        self.chat_history.append({
            "speaker": "User",
            "message": user_msg,
            "timestamp": datetime.now().isoformat()
        })
        self.chat_history.append({
            "speaker": "Bot",
            "message": bot_msg,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_all_user_text(self):
        """Combine all user messages for sentiment analysis"""
        return " ".join(self.user_messages)
    
    def get_conversation_length(self):
        """Return total number of exchanges"""
        return len(self.user_messages)
    
    def save_to_file(self):
        """Save complete conversation to text file"""
        try:
            with open(self.conversation_file, "w", encoding="utf-8") as f:
                f.write("="*70 + "\n")
                f.write("INTELLIGENT CHATBOT CONVERSATION HISTORY\n")
                f.write(f"Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {len(self.user_messages)} messages\n")
                f.write("="*70 + "\n\n")
                
                for item in self.chat_history:
                    speaker = item["speaker"]
                    message = item["message"]
                    f.write(f"[{speaker}] {message}\n\n")
            
            print(f"✅ Conversation saved to: {self.conversation_file}")
        except Exception as e:
            print(f"⚠️  Could not save conversation: {e}")


# ============================================================================
# PART 3: GEMINI AI CONNECTOR (With Auto Model Detection)
# ============================================================================

class GeminiChatbot:
    """
    Connects to Google Gemini AI with automatic model detection
    """
    
    def __init__(self):
        """Initialize with best available model"""
        # Auto-detect best model
        best_model = ModelDetector.get_best_model()
        
        if not best_model:
            print("❌ ERROR: No Gemini models available!")
            print("\nPossible causes:")
            print("1. API key doesn't have access to Generative AI")
            print("2. API key is invalid")
            print("3. API hasn't been enabled yet")
            print("\nSolution:")
            print("1. Go to: https://makersuite.google.com/app/apikey")
            print("2. Delete current key and create NEW one")
            print("3. Update .env file")
            print("4. Run this script again")
            exit()
        
        print(f"✅ Using model: {best_model}")
        self.model = genai.GenerativeModel(best_model)
        self.conversation_history = []
        
        self.system_prompt = """You are a helpful, friendly, and intelligent chatbot assistant.

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
    
    def get_response(self, user_message):
        """Generate intelligent response using Gemini AI"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "parts": [user_message]
            })
            
            # Build prompt with conversation context
            full_prompt = self.system_prompt + "\n\n"
            
            # Add recent conversation for context
            if len(self.conversation_history) > 1:
                recent_history = self.conversation_history[-10:]
                for item in recent_history:
                    if item["role"] == "user":
                        full_prompt += f"User: {item['parts'][0]}\n"
                    else:
                        full_prompt += f"Assistant: {item['parts'][0]}\n"
            
            # Add current user message
            full_prompt += f"User: {user_message}\nAssistant:"
            
            # Get response from Gemini
            response = self.model.generate_content(full_prompt)
            bot_response = response.text.strip()
            
            # Add response to history
            self.conversation_history.append({
                "role": "model",
                "parts": [bot_response]
            })
            
            return bot_response
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"⚠️  {error_msg}")
            return "I apologize, but I encountered an error. Please try again."


# ============================================================================
# PART 4: SENTIMENT ANALYZER
# ============================================================================

class SentimentAnalyzer:
    """Analyzes sentiment using TextBlob"""
    
    @staticmethod
    def analyze(text):
        """Analyze sentiment of text"""
        if not text or len(text.strip()) == 0:
            return 0.0, 0.0
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return polarity, subjectivity
    
    @staticmethod
    def classify(polarity):
        """Classify sentiment"""
        if polarity > 0.6:
            return "VERY POSITIVE", "😄"
        elif polarity > 0.2:
            return "POSITIVE", "🙂"
        elif polarity >= -0.2:
            return "NEUTRAL", "😐"
        elif polarity < -0.6:
            return "VERY NEGATIVE", "😢"
        else:
            return "NEGATIVE", "😟"
    
    @staticmethod
    def generate_insight(polarity):
        """Generate insight based on sentiment"""
        classification, _ = SentimentAnalyzer.classify(polarity)
        
        insights = {
            "VERY POSITIVE": "Your conversation is filled with very positive and enthusiastic energy!",
            "POSITIVE": "Your conversation maintains a positive and satisfied tone overall.",
            "NEUTRAL": "Your conversation is balanced with mixed perspectives.",
            "NEGATIVE": "Your conversation expresses some concerns or dissatisfaction.",
            "VERY NEGATIVE": "Your conversation conveys strong dissatisfaction or frustration."
        }
        
        return insights.get(classification, "Analysis unclear.")


# ============================================================================
# PART 5: MAIN CHATBOT APPLICATION
# ============================================================================

class EnhancedChatbotTier1:
    """Main chatbot application with auto model detection"""
    
    def __init__(self):
        """Initialize all components"""
        print("🔄 Initializing Intelligent Chatbot...")
        print("   • Auto-detecting Gemini model")
        print("   • Setting up conversation manager")
        print("   • Preparing sentiment analyzer")
        print()
        
        self.gemini = GeminiChatbot()
        self.manager = ConversationManager()
        self.analyzer = SentimentAnalyzer()
        self.is_running = True
    
    def display_welcome(self):
        """Display welcome banner"""
        print("\n" + "="*70)
        print("  INTELLIGENT CHATBOT WITH GEMINI AI & SENTIMENT ANALYSIS")
        print("="*70)
        print("\n🤖 Welcome! I'm an AI-powered chatbot using Google Gemini.")
        print("\n✨ Features:")
        print("   • Intelligent, context-aware responses")
        print("   • Full conversation memory")
        print("   • Real-time sentiment analysis")
        print("   • Conversation history saved to file")
        print("\n💬 How to use:")
        print("   • Chat naturally with me")
        print("   • I'll remember what you said")
        print("   • Type 'quit' or 'bye' when done")
        print("-"*70 + "\n")
    
    def process_message(self, user_input):
        """Process user message and get AI response"""
        print(f"\n🤔 Thinking...", end="", flush=True)
        
        # Get response from Gemini AI
        bot_response = self.gemini.get_response(user_input)
        
        # Store in conversation manager
        self.manager.add_exchange(user_input, bot_response)
        
        # Display response
        print(f"\r✅ Response ready!  \n")
        print(f"Bot: {bot_response}\n")
    
    def display_analysis(self):
        """Display sentiment analysis"""
        
        print("\n" + "="*70)
        print("  CONVERSATION SENTIMENT ANALYSIS")
        print("="*70 + "\n")
        
        all_text = self.manager.get_all_user_text()
        message_count = self.manager.get_conversation_length()
        
        if message_count == 0:
            print("No messages to analyze!")
            return
        
        # Analyze sentiment
        polarity, subjectivity = self.analyzer.analyze(all_text)
        classification, emoji = self.analyzer.classify(polarity)
        insight = self.analyzer.generate_insight(polarity)
        
        # Display results
        print(f"📊 OVERALL SENTIMENT ANALYSIS")
        print(f"   Total Messages: {message_count}")
        print(f"   Sentiment Score: {polarity:.2f} (range: -1.0 to +1.0)")
        print(f"   Classification: {emoji} {classification}")
        print(f"   Subjectivity: {subjectivity:.2f}")
        print()
        print(f"💭 INSIGHT:")
        print(f"   {insight}")
        print()
        
        # Show recent messages
        print(f"📝 YOUR RECENT MESSAGES:")
        recent_count = min(5, len(self.manager.user_messages))
        for i, msg in enumerate(self.manager.user_messages[-recent_count:], 1):
            display_msg = msg[:60] + "..." if len(msg) > 60 else msg
            print(f"   {i}. {display_msg}")
        
        if len(self.manager.user_messages) > recent_count:
            remaining = len(self.manager.user_messages) - recent_count
            print(f"   ... and {remaining} more messages")
        
        print("\n" + "="*70 + "\n")
    
    def should_quit(self, user_input):
        """Check if user wants to quit"""
        quit_words = ["quit", "bye", "goodbye", "exit", "leave", "q"]
        return any(word in user_input.lower() for word in quit_words)
    
    def run(self):
        """Main chatbot loop"""
        self.display_welcome()
        
        try:
            while self.is_running:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    print("Please say something!\n")
                    continue
                
                if self.should_quit(user_input):
                    self.is_running = False
                    break
                
                self.process_message(user_input)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Chatbot interrupted by user.")
        
        finally:
            self.display_analysis()
            self.manager.save_to_file()
            print("Thank you for chatting! Goodbye! 👋\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    chatbot = EnhancedChatbotTier1()
    chatbot.run()
