from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

print("Starting...")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OWLCHAT_API_KEY"),
    base_url=os.getenv("OWLCHAT_BASE_URL")
)

print("Client configured...")

# Read and encode the image
with open("test_frame.jpg", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

print("Image loaded, sending to AI...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": """Analyze this person's face and hair. Please provide:
1. Face shape
2. Hair texture
3. Hair density
4. Hairline shape
5. Overall hair growth direction
6. Any notable features relevant to hairstyling"""
                }
            ]
        }
    ]
)

print("Response received!")
print(response.choices[0].message.content)