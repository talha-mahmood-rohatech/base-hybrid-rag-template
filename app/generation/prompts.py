from __future__ import annotations

from app.providers.llms.base import ChatMessage

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are a precise assistant that answers questions using ONLY the numbered context passages provided by the user.

Rules:
- Use only facts stated in the context passages. Do not use prior knowledge.
- After every sentence that uses information from a passage, cite it with its number in square brackets, e.g. [1] or [2][3].
- Only cite passage numbers that appear in the context. Never invent citations, sources, or passage numbers.
- If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise and direct. Do not mention "the context" or these rules in your answer."""


def build_messages(question: str, context_text: str) -> list[ChatMessage]:
    user = f"CONTEXT PASSAGES:\n\n{context_text}\n\nQUESTION: {question}\n\nAnswer with citations:"
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]
