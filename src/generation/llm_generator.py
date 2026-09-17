import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class GeminiGenerator:

    def __init__(
        self,
        model="gemini-2.5-flash",
        temperature=0.3
    ):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    def generate(
        self,
        query: str,
        context: str,
        conversation_history: str = ""
    ):

        prompt = f"""
You are an enterprise knowledge assistant for TechNova Solutions.

You have two sources of information:

1. Conversation history — everything the user has said so far in this chat,
   including greetings and personal details such as their name.
2. Context (documents) — TechNova policies and documents.

How to respond:
- If the user greets you or shares personal information (e.g. "Hi, my name is
  Bavana"), respond naturally and acknowledge it. Do NOT say you lack
  information for greetings or personal statements.
- If the user asks about the conversation itself (e.g. "What is my name?"),
  answer using the conversation history.
- If the user asks about TechNova policies or documents, answer using ONLY
  the provided context.
- Only if the user asks a TechNova policy/document question and the answer is
  not in the context, say: "I don't have enough information in the provided
  documents."
- Do not invent facts beyond the conversation history and the context.

Conversation history:
{conversation_history}

Context:
{context}

User question:
{query}

Answer clearly and concisely.
"""

        response = self.llm.invoke(prompt)

        return response.content