"""
Defines system prompts used in the research.
"""
BASE_SYS_PROMPT = """
You are a helpful AI assistant.
Generate most relevant and short search question for the document excerpt provided by the user.
Return only the generated query without quotes or trailing punctuation marks.
""".strip()
