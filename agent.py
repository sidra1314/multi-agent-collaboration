import chainlit as cl
import os
import re
import json
from agents import (
    agent,
    RunHooks,
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_tracing_disabled,
    AsyncOpenAI,
    tool,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
set_default_openai_client(external_client)
set_tracing_disabled(True)

# Initialize model
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

# Store conversation history
chat_history = []

# Notes file
NOTES_FILE = "notes.json"

# Tools
@tool
def calculator(expression: str) -> dict:
    """Calculate a math expression and return JSON."""
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        return {"status": "error", "details": "Invalid expression: only numbers and + - * / ( ) allowed"}
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"status": "success", "details": str(result)}
    except Exception as e:
        return {"status": "error", "details": f"Calculation error: {str(e)}"}

@tool
def save_note(note: str) -> dict:
    """Save a note to a file and return JSON."""
    if not note.strip():
        return {"status": "error", "details": "Note cannot be empty"}
    try:
        with open(NOTES_FILE, 'a') as f:
            json.dump({"note": note}, f)
            f.write("\n")
        return {"status": "success", "details": f"Note saved: {note}"}
    except Exception as e:
        return {"status": "error", "details": f"Error saving note: {str(e)}"}

@tool
def fix_grammar(text: str) -> dict:
    """Fix grammar using a sub-agent and return JSON."""
    if not text.strip():
        return {"status": "error", "details": "Text cannot be empty"}
    try:
        response = grammar_fix_agent.invoke(text)
        return {"status": "success", "details": response}
    except Exception as e:
        return {"status": "error", "details": f"Grammar fix error: {str(e)}"}

@tool
def escalate_to_human(query: str) -> dict:
    """Escalate complex queries to a human."""
    return {"status": "escalated", "details": f"Query escalated to human: {query}"}

# Grammar sub-agent
@agent(instructions="You fix grammar mistakes.", model="gemini-1.5-flash")
def grammar_fix_agent(text: str):
    return f"[Grammar Fixed]: {text}"

# Lifecycle hooks
class LoggingHook(RunHooks):
    def on_agent_start(self, agent):
        print(f"Agent started: {agent.name}")
    def on_tool_start(self, tool_call):
        print(f"Tool started: {tool_call.name}")
    def on_tool_end(self, tool_call, result):
        print(f"Tool result: {result}")
    def on_agent_end(self, agent):
        print(f"Agent ended: {agent.name}")

# Main agent
@agent(
    tools=[calculator, save_note, fix_grammar, escalate_to_human],
    run_hooks=[LoggingHook()],
    instructions="You are a Smart Education Agent. Use tools to help with math, notes, grammar, or escalate complex queries.",
    model="gemini-2.5-flash"
)
class SmartEduAgent:
    def __init__(self):
        self.running = False

    def start(self):
        self.running = True
        print("SmartEduAgent started")

    async def process(self, task: str, context: dict = {}) -> dict:
        if not self.running:
            return {"status": "error", "details": "Agent not running"}

        # Guardrail: Check for invalid input
        if not task.strip() or any(word in task.lower() for word in ["hack", "attack"]):
            return {"status": "error", "details": "Invalid or unsafe input"}

        # Save to history
        global chat_history
        chat_history.append({"user": task})

        # Classify task
        task_lower = task.lower()
        if "calculate" in task_lower or re.search(r'[\d+\-*/()]', task):
            category = "calculator"
        elif "note" in task_lower:
            category = "save_note"
        elif "grammar" in task_lower:
            category = "fix_grammar"
        elif "complex" in task_lower:
            category = "escalate_to_human"
        else:
            category = "general"

        # Process task
        if category == "calculator":
            expression = task.replace("calculate", "").strip()
            result = calculator(expression)
        elif category == "save_note":
            note = task.replace("note", "").strip()
            result = save_note(note)
        elif category == "fix_grammar":
            text = task.replace("grammar", "").strip()
            result = fix_grammar(text)
        elif category == "escalate_to_human":
            result = escalate_to_human(task)
        else:
            try:
                response = await external_client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=[
                        {"role": "system", "content": "Answer in JSON: {'status': 'success', 'details': 'response text'}"},
                        {"role": "user", "content": f"Context: {json.dumps(chat_history[-2:])} Task: {task}"}
                    ]
                )
                result = json.loads(response.choices[0].message.content)
            except Exception as e:
                result = {"status": "error", "details": f"Error: {str(e)}"}

        # Save response to history
        chat_history.append({"bot": result})
        return result

    def stop(self):
        self.running = False
        print("SmartEduAgent stopped")

# Create agent instance
smart_edu_agent = SmartEduAgent()

# Chainlit frontend
@cl.on_chat_start
async def start():
    smart_edu_agent.start()
    await cl.Message(content="Welcome to Smart Education Agent! I can help with math, notes, grammar, or escalate complex queries.").send()

@cl.on_message
async def handle_message(msg: cl.Message):
    task = msg.content
    result = await smart_edu_agent.process(task)
    await cl.Message(content=json.dumps(result, indent=2)).send()

@cl.on_chat_end
async def end():
    smart_edu_agent.stop()
    await cl.Message(content="Goodbye!").send()
        
    
        







