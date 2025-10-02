import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.agents import ChatCompletionAgent, HandoffOrchestration, OrchestrationHandoffs
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent
from agents import disconnect_mcp


# ==============================
# Runtime singleton
# ==============================
runtime = InProcessRuntime()
runtime_started = False
return_message = ""

def start_runtime():
    global runtime_started
    if not runtime_started:
        runtime.start()
        runtime_started = True

async def stop_runtime():
    global runtime_started
    if runtime_started:
        await disconnect_mcp()
        await runtime.stop_when_idle()
        runtime_started = False

# ==============================
# Chat sessions (per user)
# ==============================
chat_sessions = {}  # {session_id: {"history": ChatHistory(), "orchestration": HandoffOrchestration}}

def get_chat_history(session_id: str) -> ChatHistory:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"history": ChatHistory(), "orchestration": None}
    return chat_sessions[session_id]["history"]


# ==============================
# Agent response callback
# ==============================
def agent_response_callback(message: ChatMessageContent):
    # Print the main message only once
    print(message)
    if message.content.strip():
        print(f"{message.name}: {message.content}")
        print("from agent_response_callback")
        if hasattr(runtime, 'last_agent_question'):
            runtime.last_agent_question = message.content
        else:
            setattr(runtime, 'last_agent_question', message.content)

    # Print only tool calls/results
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"🔧 Calling function '{item.name}' with arguments: {item.arguments}")
        elif isinstance(item, FunctionResultContent):
            print(f"✅ Result from '{item.name}': {item.result}")

    # return return_message


def last_n_prompt(history: ChatHistory, n: int = 5) -> str:
    # Get last n messages
    last_messages = history[-n:] if len(history) > n else history
    
    # Build a temporary ChatHistory with just those
    temp_history = ChatHistory()
    for msg in last_messages:
        temp_history.add_message(msg)
    
    # Convert to prompt string
    return temp_history.to_prompt()