import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

with open("./sorting_algorithms.txt", "r") as f:
    file_content = f.read()

    response = client.messages.create(
        model="claude-opus-4-1-20250805",  # or any supported model
        max_tokens=1024,
        messages=[
            {
                "role": "user", 
                "content": f"Here's a document:\n\n{file_content}\n\nPlease summarize this in 3 bullet points."
            }
        ],
    )

    print(response.content)
