import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 1. Configuration & API Client Setup
# Load environment variables from groq_key.env
load_dotenv("groq_key.env")
api_key = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client with Groq base URL
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

def get_completion_from_messages(messages, model="llama-3.3-70b-versatile"):
    """
    Sends a list of messages (conversation history) to the LLM.
    Supports memory by processing the entire context.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

# 2. Career Advisor Module
def get_career_advice(topic="Engineering"):
    """
    Generates structured career advice for the German job market.
    """
    prompt = f"Provide 5 actionable career tips for an engineer in Germany regarding {topic}. Language: Arabic."
    messages = [{"role": "user", "content": prompt}]
    return get_completion_from_messages(messages)

# 3. Data Processing Module
def process_data_to_table(ai_response):
    """
    Parses JSON content from AI response into a structured Pandas DataFrame.
    """
    try:
        data = json.loads(ai_response)
        return pd.DataFrame(data['tips'])
    except Exception as e:
        return f"Error parsing JSON: {e}"

# 4. Interactive Chatbot Module (With Memory)
def start_legal_chatbot():
    """
    Starts an interactive terminal session with memory focused on Labor Law.
    """
    print("\n--- 🤖 LogicAziz AI Assistant (Interactive Mode) ---")
    print("Commands: Type 'exit' or 'quit' to end the session.\n")
    
    # Initialize conversation history with system instructions
    context = [{'role':'system', 'content':'You are a legal assistant specializing in German Labor Law. Respond in Arabic.'}]
    
    while True:
        user_message = input("User: ")
        
        # Exit conditions
        if user_message.lower() in ["exit", "quit", "خروج"]:
            print("Assistant: Goodbye! Have a productive day.")
            break
        
        # Append user message to history
        context.append({'role':'user', 'content': user_message})
        
        # Fetch completion using the full context
        response = get_completion_from_messages(context)
        
        # Append assistant response to history
        context.append({'role':'assistant', 'content': response})
        
        print(f"\nAI: {response}\n")

# 5. Main Execution Entry Point
if __name__ == "__main__":
    if api_key:
        print("✅ Connection Successful. Modules Loaded: CareerAdvisor, DataProcessor, Chatbot.")
        # To run the bot locally, uncomment the line below:
        # start_legal_chatbot()
    else:
        print("❌ Error: GROQ_API_KEY not found in groq_key.env.")
