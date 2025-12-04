from flask import Flask, request, jsonify
from tier2_chatbot import Tier2ChatbotApplication

app = Flask(_name_)
chatbot = Tier2ChatbotApplication()

@app.post("/chat")
def chat():
    user_message = request.json["message"]
    bot_reply = chatbot.gemini.get_response(user_message)
    return jsonify({"response": bot_reply})

if _name_ == "_main_":
    app.run(debug=True)