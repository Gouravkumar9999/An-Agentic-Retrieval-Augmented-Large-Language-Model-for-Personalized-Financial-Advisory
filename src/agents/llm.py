from groq import Groq
import os

client = Groq(api_key="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")
def generate_response(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[
                {"role": "system", "content": "You are a helpful financial advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=800   
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"
    

    