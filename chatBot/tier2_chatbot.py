"""
ENHANCED TIER 2: Advanced Intelligent Chatbot with Real-Time Sentiment Analysis
==================================================================================
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
"""

import os
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from textblob import TextBlob
from statistics import mean, stdev

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
# PART 1: MODEL DETECTOR (Auto-Detect Available Models)
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
        
        for model in preferred_models:
            if model in available:
                return model
        
        if available:
            return available[0]
        
        return None


# ============================================================================
# PART 2: ADVANCED SENTIMENT ANALYZER (Real-Time Per-Message)
# ============================================================================

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analysis with:
    - Per-message scoring
    - Trend detection
    - Detailed statistics
    - Emotional classification
    """
    
    def __init__(self):
        self.sentiment_history = []
        self.emotional_keywords = {
            'positive': ['love', 'great', 'excellent', 'happy', 'good', 'wonderful', 'amazing', 'fantastic', 'awesome'],
            'negative': ['hate', 'bad', 'terrible', 'sad', 'awful', 'horrible', 'worst', 'disgusting', 'angry'],
            'neutral': ['okay', 'fine', 'alright', 'average', 'normal']
        }
    
    def analyze_message(self, text):
        """Analyze single message and return detailed sentiment"""
        if not text or len(text.strip()) == 0:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'classification': 'NEUTRAL',
                'emoji': '😐',
                'confidence': 0.0,
                'emotional_tone': 'neutral'
            }
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        classification, emoji = self._classify_sentiment(polarity)
        emotional_tone = self._detect_emotional_tone(text)
        confidence = self._calculate_confidence(polarity, subjectivity)
        
        sentiment_data = {
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'classification': classification,
            'emoji': emoji,
            'confidence': round(confidence, 3),
            'emotional_tone': emotional_tone,
            'timestamp': datetime.now().isoformat()
        }
        
        self.sentiment_history.append(sentiment_data)
        return sentiment_data
    
    def _classify_sentiment(self, polarity):
        """Classify sentiment based on polarity"""
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
    
    def _detect_emotional_tone(self, text):
        """Detect specific emotional tone"""
        text_lower = text.lower()
        
        for emotion, keywords in self.emotional_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return "neutral"
    
    def _calculate_confidence(self, polarity, subjectivity):
        """Calculate confidence in sentiment analysis"""
        # Higher confidence if polarity is strong and subjectivity is high
        polarity_confidence = abs(polarity)
        subjectivity_confidence = min(subjectivity * 1.5, 1.0)
        
        return mean([polarity_confidence, subjectivity_confidence])
    
    def get_trend(self):
        """Detect mood trend across conversation"""
        if len(self.sentiment_history) < 2:
            return "STABLE", "📊"
        
        recent_scores = [s['polarity'] for s in self.sentiment_history[-5:]]
        older_scores = [s['polarity'] for s in self.sentiment_history[:max(1, len(self.sentiment_history)-5)]]
        
        recent_avg = mean(recent_scores)
        older_avg = mean(older_scores)
        
        difference = recent_avg - older_avg
        
        if difference > 0.1:
            return "IMPROVING", "📈"
        elif difference < -0.1:
            return "DECLINING", "📉"
        else:
            return "STABLE", "📊"
    
    def get_statistics(self):
        """Generate detailed sentiment statistics"""
        if not self.sentiment_history:
            return None
        
        polarities = [s['polarity'] for s in self.sentiment_history]
        subjectivities = [s['subjectivity'] for s in self.sentiment_history]
        classifications = [s['classification'] for s in self.sentiment_history]
        emotional_tones = [s['emotional_tone'] for s in self.sentiment_history]
        
        stats = {
            'total_messages': len(self.sentiment_history),
            'average_polarity': round(mean(polarities), 3),
            'polarity_range': (round(min(polarities), 3), round(max(polarities), 3)),
            'polarity_std_dev': round(stdev(polarities), 3) if len(polarities) > 1 else 0,
            'average_subjectivity': round(mean(subjectivities), 3),
            'sentiment_distribution': self._count_classifications(classifications),
            'emotional_tone_distribution': self._count_emotional_tones(emotional_tones),
            'trend': self.get_trend()[0],
            'trend_emoji': self.get_trend()[1],
            'overall_mood': self._determine_overall_mood(mean(polarities))
        }
        
        return stats
    
    def _count_classifications(self, classifications):
        """Count sentiment classifications"""
        counts = {}
        for cls in classifications:
            counts[cls] = counts.get(cls, 0) + 1
        return counts
    
    def _count_emotional_tones(self, tones):
        """Count emotional tones"""
        counts = {}
        for tone in tones:
            counts[tone] = counts.get(tone, 0) + 1
        return counts
    
    def _determine_overall_mood(self, polarity):
        """Determine overall mood description"""
        if polarity > 0.5:
            return "Very Positive & Enthusiastic"
        elif polarity > 0.1:
            return "Positive & Satisfied"
        elif polarity >= -0.1:
            return "Neutral & Balanced"
        elif polarity > -0.5:
            return "Negative & Concerned"
        else:
            return "Very Negative & Frustrated"


# ============================================================================
# PART 3: ADVANCED CONVERSATION MANAGER
# ============================================================================

class AdvancedConversationManager:
    """
    Manages conversation with detailed tracking:
    - Per-message sentiment
    - Message metadata
    - Conversation analytics
    - Export capabilities
    """
    
    def __init__(self):
        self.messages = []
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.start_time = datetime.now()
        self.conversation_file = "tier2_conversation_history.txt"
        self.json_export = "tier2_conversation_export.json"
        self.csv_export = "tier2_conversation_export.csv"
    
    def add_user_message(self, message):
        """Add user message with sentiment analysis"""
        sentiment = self.sentiment_analyzer.analyze_message(message)
        
        self.messages.append({
            'speaker': 'User',
            'message': message,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_bot_message(self, message):
        """Add bot message with sentiment analysis"""
        sentiment = self.sentiment_analyzer.analyze_message(message)
        
        self.messages.append({
            'speaker': 'Bot',
            'message': message,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_conversation_summary(self):
        """Get summary of entire conversation"""
        user_messages = [m for m in self.messages if m['speaker'] == 'User']
        bot_messages = [m for m in self.messages if m['speaker'] == 'Bot']
        
        return {
            'total_exchanges': len(user_messages),
            'total_messages': len(self.messages),
            'user_messages': len(user_messages),
            'bot_messages': len(bot_messages),
            'duration': str(datetime.now() - self.start_time).split('.')[0]
        }
    
    def get_detailed_breakdown(self):
        """Get message-by-message breakdown"""
        breakdown = []
        for i, msg in enumerate(self.messages, 1):
            breakdown.append({
                'number': i,
                'speaker': msg['speaker'],
                'message': msg['message'][:100] + "..." if len(msg['message']) > 100 else msg['message'],
                'sentiment': msg['sentiment']['classification'],
                'emoji': msg['sentiment']['emoji'],
                'polarity': msg['sentiment']['polarity'],
                'confidence': msg['sentiment']['confidence']
            })
        return breakdown
    
    def save_to_text_file(self):
        """Save conversation to readable text file"""
        try:
            with open(self.conversation_file, "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write("TIER 2: INTELLIGENT CHATBOT WITH REAL-TIME SENTIMENT ANALYSIS\n")
                f.write(f"Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {str(datetime.now() - self.start_time).split('.')[0]}\n")
                f.write("="*80 + "\n\n")
                
                for msg in self.messages:
                    speaker = msg['speaker']
                    message = msg['message']
                    sentiment = msg['sentiment']
                    
                    f.write(f"[{speaker}] {sentiment['emoji']} ({sentiment['classification']})\n")
                    f.write(f"{message}\n")
                    f.write(f"  Polarity: {sentiment['polarity']}, Confidence: {sentiment['confidence']}\n\n")
            
            print(f"✅ Text file saved to: {self.conversation_file}")
        except Exception as e:
            print(f"⚠️  Could not save text file: {e}")
    
    def export_to_json(self):
        """Export conversation to JSON"""
        try:
            export_data = {
                'metadata': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration': str(datetime.now() - self.start_time).split('.')[0],
                    'total_messages': len(self.messages)
                },
                'summary': self.get_conversation_summary(),
                'statistics': self.sentiment_analyzer.get_statistics(),
                'messages': self.messages
            }
            
            with open(self.json_export, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON export saved to: {self.json_export}")
        except Exception as e:
            print(f"⚠️  Could not save JSON export: {e}")
    
    def export_to_csv(self):
        """Export sentiment data to CSV"""
        try:
            with open(self.csv_export, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'Message_Number',
                    'Speaker',
                    'Message',
                    'Sentiment_Classification',
                    'Polarity',
                    'Subjectivity',
                    'Confidence',
                    'Emotional_Tone',
                    'Timestamp'
                ])
                
                # Write data
                for i, msg in enumerate(self.messages, 1):
                    sentiment = msg['sentiment']
                    writer.writerow([
                        i,
                        msg['speaker'],
                        msg['message'][:100],
                        sentiment['classification'],
                        sentiment['polarity'],
                        sentiment['subjectivity'],
                        sentiment['confidence'],
                        sentiment['emotional_tone'],
                        msg['timestamp']
                    ])
            
            print(f"✅ CSV export saved to: {self.csv_export}")
        except Exception as e:
            print(f"⚠️  Could not save CSV export: {e}")


# ============================================================================
# PART 4: GEMINI AI CONNECTOR (With Auto Model Detection)
# ============================================================================

class GeminiChatbot:
    """Connects to Gemini AI with automatic model detection"""
    
    def __init__(self):
        """Initialize with best available model"""
        best_model = ModelDetector.get_best_model()
        
        if not best_model:
            print("❌ ERROR: No Gemini models available!")
            print("\nSolution: Get new API key from https://makersuite.google.com/app/apikey")
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
            self.conversation_history.append({
                "role": "user",
                "parts": [user_message]
            })
            
            full_prompt = self.system_prompt + "\n\n"
            
            if len(self.conversation_history) > 1:
                recent_history = self.conversation_history[-10:]
                for item in recent_history:
                    if item["role"] == "user":
                        full_prompt += f"User: {item['parts'][0]}\n"
                    else:
                        full_prompt += f"Assistant: {item['parts'][0]}\n"
            
            full_prompt += f"User: {user_message}\nAssistant:"
            
            response = self.model.generate_content(full_prompt)
            bot_response = response.text.strip()
            
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
# PART 5: TIER 2 MAIN APPLICATION
# ============================================================================

class Tier2ChatbotApplication:
    """
    Advanced Tier 2 Chatbot with Real-Time Sentiment Analysis
    """
    
    def __init__(self):
        """Initialize all components"""
        print("🔄 Initializing Tier 2 Chatbot with Real-Time Analysis...")
        print("   • Auto-detecting Gemini model")
        print("   • Setting up advanced conversation manager")
        print("   • Initializing real-time sentiment analyzer")
        print()
        
        self.gemini = GeminiChatbot()
        self.manager = AdvancedConversationManager()
        self.is_running = True
    
    def display_welcome(self):
        """Display welcome banner"""
        print("\n" + "="*80)
        print("  TIER 2: INTELLIGENT CHATBOT WITH REAL-TIME SENTIMENT ANALYSIS")
        print("="*80)
        print("\n🤖 Welcome to Advanced Chatbot! I'm using Google Gemini AI.")
        print("\n✨ Tier 2 Features:")
        print("   • Real-time sentiment analysis for each message")
        print("   • Mood trend tracking across conversation")
        print("   • Detailed sentiment statistics")
        print("   • Full conversation memory")
        print("   • Sentiment breakdown visualization")
        print("   • Export to JSON and CSV")
        print("\n💬 How to use:")
        print("   • Chat naturally with me")
        print("   • See live sentiment analysis for each message")
        print("   • Type 'quit' when done to see detailed analytics")
        print("-"*80 + "\n")
    
    def process_message(self, user_input):
        """Process message with real-time sentiment analysis"""
        print(f"\n🤔 Analyzing...", end="", flush=True)
        
        # Add user message with sentiment
        self.manager.add_user_message(user_input)
        user_sentiment = self.manager.sentiment_analyzer.sentiment_history[-1]
        
        # Display real-time user sentiment
        print(f"\r✅ User sentiment: {user_sentiment['emoji']} {user_sentiment['classification']} " + 
              f"(Polarity: {user_sentiment['polarity']}, Confidence: {user_sentiment['confidence']})")
        
        # Get bot response
        print(f"🤖 Thinking...", end="", flush=True)
        bot_response = self.gemini.get_response(user_input)
        
        # Add bot message with sentiment
        self.manager.add_bot_message(bot_response)
        bot_sentiment = self.manager.sentiment_analyzer.sentiment_history[-1]
        
        # Display response with sentiment
        print(f"\r✅ Bot response ready!")
        print(f"\nBot: {bot_response}")
        print(f"\n   {bot_sentiment['emoji']} Sentiment: {bot_sentiment['classification']} | " +
              f"Polarity: {bot_sentiment['polarity']} | Confidence: {bot_sentiment['confidence']}")
        
        # Show trend
        trend, trend_emoji = self.manager.sentiment_analyzer.get_trend()
        print(f"   {trend_emoji} Trend: {trend}")
    
    def display_analytics(self):
        """Display detailed sentiment analytics"""
        print("\n\n" + "="*80)
        print("  TIER 2: DETAILED SENTIMENT ANALYSIS REPORT")
        print("="*80 + "\n")
        
        summary = self.manager.get_conversation_summary()
        stats = self.manager.sentiment_analyzer.get_statistics()
        breakdown = self.manager.get_detailed_breakdown()
        
        if not stats:
            print("No messages to analyze!")
            return
        
        # Conversation Summary
        print(f"📊 CONVERSATION SUMMARY")
        print(f"   Total Exchanges: {summary['total_exchanges']}")
        print(f"   Total Messages: {summary['total_messages']}")
        print(f"   Duration: {summary['duration']}")
        print()
        
        # Sentiment Statistics
        print(f"📈 SENTIMENT STATISTICS")
        print(f"   Average Polarity: {stats['average_polarity']} (range: -1 to +1)")
        print(f"   Polarity Range: {stats['polarity_range'][0]} to {stats['polarity_range'][1]}")
        print(f"   Standard Deviation: {stats['polarity_std_dev']}")
        print(f"   Average Subjectivity: {stats['average_subjectivity']}")
        print()
        
        # Sentiment Distribution
        print(f"📊 SENTIMENT DISTRIBUTION")
        for classification, count in stats['sentiment_distribution'].items():
            percentage = (count / stats['total_messages']) * 100
            print(f"   {classification}: {count} messages ({percentage:.1f}%)")
        print()
        
        # Emotional Tone Distribution
        print(f"❤️ EMOTIONAL TONE DISTRIBUTION")
        for tone, count in stats['emotional_tone_distribution'].items():
            percentage = (count / stats['total_messages']) * 100
            print(f"   {tone.upper()}: {count} messages ({percentage:.1f}%)")
        print()
        
        # Overall Analysis
        print(f"💭 OVERALL ANALYSIS")
        print(f"   Overall Mood: {stats['overall_mood']}")
        print(f"   Trend: {stats['trend_emoji']} {stats['trend']}")
        print()
        
        # Message Breakdown (First 10)
        print(f"📝 MESSAGE BREAKDOWN (First 10)")
        for msg in breakdown[:10]:
            print(f"   {msg['number']:2d}. [{msg['speaker']:4s}] {msg['emoji']} " +
                  f"{msg['sentiment']:15s} | Polarity: {msg['polarity']:6.3f}")
        
        if len(breakdown) > 10:
            print(f"   ... and {len(breakdown) - 10} more messages")
        
        print("\n" + "="*80 + "\n")
    
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
                    user_input = input("\nYou: ").strip()
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
            self.display_analytics()
            self.manager.save_to_text_file()
            self.manager.export_to_json()
            self.manager.export_to_csv()
            print("📁 Files saved:")
            print(f"   • {self.manager.conversation_file}")
            print(f"   • {self.manager.json_export}")
            print(f"   • {self.manager.csv_export}")
            print("\nThank you for using Tier 2 Chatbot! Goodbye! 👋\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    chatbot = Tier2ChatbotApplication()
    chatbot.run()
