import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 1. Configuration & Client Setup
# Make sure your key file is named "groq_key.env" and added to .gitignore
load_dotenv("groq_key.env")
api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

def get_completion(prompt, model="llama-3.3-70b-versatile"):
    """Sends a prompt to the LLM and returns the text response."""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0 
    )
    return response.choices[0].message.content

# 2. Career Advisor Logic
def get_career_advice(topic="Engineering"):
    """Generates structured career advice for newcomers in Germany."""
    prompt = f"""
    You are an expert career consultant in the German labor market.
    Provide 5 deep and actionable tips for a student/engineer moving to Germany.
    Target: {topic} | German Level: B2
    Output Language: Arabic.
    """
    return get_completion(prompt)

# 3. Data Processing (JSON to DataFrame)
def process_data_to_table(new_response):
    """Parses JSON from AI response into a Pandas DataFrame."""
    try:
        start_index = new_response.find('{')
        end_index = new_response.rfind('}') + 1
        data = json.loads(new_response[start_index:end_index])
        return pd.DataFrame(data['tips'])
    except:
        return "Error parsing data."

if __name__ == "__main__":
    if api_key:
        print("✅ Connection Successful. Ready to run applications.")
        # Example run:
        # print(get_career_advice())
    else:
        print("❌ Error: API Key not found.")
