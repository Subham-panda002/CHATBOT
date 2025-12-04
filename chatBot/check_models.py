# Complete Guide: Fix "Model Not Found" Error

## 🔍 Step 1: Check Available Models

#Create a new file called `check_models.py`:

#```python
"""
Check which Gemini models your API key has access to
Run this to see all available models
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')

if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY not found in .env")
    exit()

# Configure Gemini
genai.configure(api_key=API_KEY)

print("🔍 Checking available models...\n")
print("="*60)

# List all available models
available_models = []
try:
    for model in genai.list_models():
        # Check if model supports generateContent
        if 'generateContent' in model.supported_generation_methods:
            available_models.append(model.name)
            print(f"✅ {model.name}")
            print(f"   Display Name: {model.display_name}")
            print(f"   Supported Methods: {model.supported_generation_methods}")
            print()

except Exception as e:
    print(f"❌ Error listing models: {e}")
    print("\nTroubleshooting:")
    print("1. Check if API key is valid")
    print("2. Verify API key has Generative AI enabled")
    print("3. Try generating a new API key")
    exit()

print("="*60)
print(f"\n✅ Total available models: {len(available_models)}\n")

if available_models:
    print("Use one of these in your code:")
    for i, model_name in enumerate(available_models, 1):
        # Extract model name without full path
        short_name = model_name.replace('models/', '')
        print(f"{i}. '{short_name}'")
        print(f"   (Full: {model_name})\n")
else:
    print("❌ No models found!")
    print("Your API key might not have access to Generative AI.")
    print("\nSolution:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Delete your current key")
    print("3. Create a NEW key")
    print("4. Update .env file")
    print("5. Run this script again")
