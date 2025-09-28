import os
import uuid
import asyncio
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from semantic_kernel.agents import HandoffOrchestration
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from agents import get_agents, create_handoffs
from util import start_runtime, stop_runtime, agent_response_callback, runtime

app = FastAPI()
load_dotenv()



class StartSessionResponse(BaseModel):
    session_id: str
    

class SendMessageRequest(BaseModel):
    session_id: str
    message: str

class SendMessageResponse(BaseModel):
    status: str

class GetResponseResponse(BaseModel):
    status: str
    message: str | None = None

# Store chat session data per user
chat_sessions: dict[str, dict] = {}


@app.on_event("startup")
async def startup_event():
    """Start the global runtime once."""
    start_runtime()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop runtime gracefully."""
    await stop_runtime()



def make_human_response_function(session_id: str):
    async def _human_response():
        session = chat_sessions[session_id]
        question = runtime.last_agent_question if hasattr(runtime, 'last_agent_question') else "Agent needs input."
        session["unpolled_agent_messages"].append({
            "role": "agent",
            "content": question,
            "type": "await_human"
        })
        while not session["pending_human_input"]:
            await asyncio.sleep(0.1)
        user_input = session["pending_human_input"].pop(0)
        return ChatMessageContent(role=AuthorRole.USER, content=user_input)
    return _human_response

async def process_agent_message(session_id: str, user_message: str):
    session = chat_sessions[session_id]
    history = session["history"]
    orchestration = session["orchestration"]
    orchestration_result = await orchestration.invoke(
        task=user_message,
        runtime=runtime,
    )
    value = await orchestration_result.get()
    # Save agent reply to history and queue for polling
    history.append({"role": "assistant", "content": value})
    session["unpolled_agent_messages"].append({
        "role": "assistant",
        "content": value,
        "type": "agent_response"
    })

@app.post("/start_session", response_model=StartSessionResponse)
async def start_session():
    """Start a new chat session and return a session_id."""
    session_id = str(uuid.uuid4())
    agents = await get_agents()
    handoffs = create_handoffs(agents)
    orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        agent_response_callback=agent_response_callback,
        human_response_function=make_human_response_function(session_id)
    )
    chat_sessions[session_id] = {
        "history": [],
        "orchestration": orchestration,
        "unpolled_agent_messages": [],
        "pending_human_input": [],
    }
    return {"session_id": session_id}

@app.post("/send_message", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id
    user_message = request.message
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = chat_sessions[session_id]
    history = session["history"]
    # Add user message to history
    history.append({"role": "user", "content": user_message})
    # If agent is waiting for human input, provide it
    if session["unpolled_agent_messages"] and session["unpolled_agent_messages"][-1].get("type") == "await_human":
        session["pending_human_input"].append(user_message)
        return {"status": "received_human_input"}
    # Otherwise, process message and queue agent response in background
    background_tasks.add_task(process_agent_message, session_id, user_message)
    return {"status": "message_processing"}

@app.get("/get_response", response_model=GetResponseResponse)
async def get_response(session_id: str = Query(...)):
    """Poll for the next agent message for this session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = chat_sessions[session_id]
    if session["unpolled_agent_messages"]:
        msg = session["unpolled_agent_messages"].pop(0)
        return {"status": msg["type"], "message": msg["content"]}
    return {"status": "no_new_message", "message": None}




