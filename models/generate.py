

import os

from dotenv import load_dotenv
from groq import Groq


# Load .env
load_dotenv()


# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file."
    )


# Create Groq client
client = Groq(api_key=api_key)


# Groq model
MODEL_NAME = "openai/gpt-oss-20b"

def generate_answer(question, context):

    prompt = f"""
You are a question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the context,
   say: "I don't have enough information in the provided document."
4. Give a clear and concise answer.

-------------------------
CONTEXT
-------------------------

{context}

-------------------------
QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    context="Amarendra Baahubali is the crown prince and chosen king of Mahishmati, raised by Rajmata Sivagami. Known for his immense strength, bravery, and deep love for his citizens, his popularity sparks a bitter rivalry with his cousin Bhallaladeva. This jealousy leads to a tragic political conspiracy where he is betrayed and assassinated by the royal slave Kattappa. His legacy is later avenged by his son, Mahendra Baahubali (Shivudu), who returns to liberate the kingdom from tyranny."
    print(generate_answer("who is bahubali",context))