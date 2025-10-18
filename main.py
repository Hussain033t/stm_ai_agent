import uuid
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Header, Request
from semantic_kernel.agents import HandoffOrchestration
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from agents import connect_mcps, disconnect_mcps, get_agents, create_handoffs
from util import agent_response_callback, make_human_response_function, process_agent_message, chat_sessions
from semantic_kernel.agents.runtime import InProcessRuntime
from functools import partial
from datetime import datetime, timezone, timedelta
import asyncio
from contextlib import asynccontextmanager

from models import StartSessionResponse, SendMessageRequest, SendMessageResponse, GetResponseResponse


# Replace deprecated @app.on_event("startup") with lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_task = asyncio.create_task(session_cleanup_task())
    yield
    # Shutdown
    cleanup_task.cancel()

app = FastAPI(
    title="STM AI Agent API",
    description="API for STM AI Agent system",
    version="1.0.0",
    lifespan=lifespan
)

SESSION_EXPIRY_MINUTES = 20

async def session_cleanup_task():
    while True:
        now = datetime.now(timezone.utc)
        expired = []
        for session_id, session in list(chat_sessions.items()):
            created = session.get("created_at")
            if created and now - created > timedelta(minutes=SESSION_EXPIRY_MINUTES):
                expired.append(session_id)
        for session_id in expired:
            await cleanup_session(session_id)
            print(f"Session {session_id} expired and removed.")
        await asyncio.sleep(300)  # Run every 5 minutes

async def cleanup_session(session_id):
    session = chat_sessions.get(session_id)
    if session:
        # If not already marked as expired, mark as expired and return
        if not session.get("expired"):
            session["expired"] = True
            print(f"Session {session_id} marked as expired.")
            return
        # If already expired, disconnect and remove
        if "plugins" in session:
            await disconnect_mcps(session["plugins"])
        if "orchestration" in session:
            session["orchestration"] = None
    chat_sessions.pop(session_id, None)

@app.post("/start_session",
    response_model=StartSessionResponse,
    tags=["Session"],
    summary="Start a new chat session",
    description=
    """
    Start a new chat session and return a session_id.
    - **Authorization**: Bearer token for agent authentication
    - **Returns**: session_id (UUID)
    """
)
async def start_session(Authorization: str = Header(..., description="Authorization token for mcp - api access")):
    session_id = str(uuid.uuid4())
    plugins = await connect_mcps(Authorization)
    agents = await get_agents(Authorization, plugins)
    handoffs = create_handoffs(agents)
    orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        agent_response_callback=partial(agent_response_callback, session_id=session_id),
        human_response_function=make_human_response_function(session_id)
    )
    
    session_runtime = InProcessRuntime()
    session_runtime.start()
    chat_sessions[session_id] = {
        "thread": [],
        "orchestration": orchestration,
        "runtime": session_runtime,
        "unpolled_agent_messages": [],
        "pending_human_input": [],
        "created_at": datetime.now(timezone.utc),
        "plugins": plugins
    }
    
    return {"session_id": session_id}

@app.post("/send_message",
    response_model=SendMessageResponse,
    tags=["Session"],
    summary="Send a message to the agent",
    description= 
    """
    Send a message to the agent in the specified session.
    - **session_id**: Session identifier
    - **message**: User message to send
    - **Returns**: Status of message processing
    """
)
async def send_message(request: SendMessageRequest, background_tasks: BackgroundTasks):
   
    session_id = request.session_id
    user_message = request.message
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = chat_sessions[session_id]
    thread = session["thread"]
    # Add user message to thread
    thread.append(ChatMessageContent(role=AuthorRole.USER, content=user_message))

    # If agent is waiting for human input, provide it
    if session["unpolled_agent_messages"] and session["unpolled_agent_messages"][-1].get("type") == "await_human":
        session["pending_human_input"].append(user_message)
        return {"status": "received_human_input"}
    
    # Otherwise, process message and queue agent response in background
    background_tasks.add_task(process_agent_message, session_id, user_message)
    return {"status": "message_processing"}

@app.get("/get_response",
    response_model=GetResponseResponse,
    tags=["Session"],
    summary="Get agent response",
    description=
    """
    Poll for the next agent message for this session.
    - **sessionId**: Session identifier to poll for agent response
    - **Returns**: Agent response or status
    """
)
async def get_response(sessionId: str = Query(..., description="Session identifier to poll for agent response")):
    
    if sessionId not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = chat_sessions[sessionId]
    if session["unpolled_agent_messages"]:
        msg = session["unpolled_agent_messages"].pop(0)
        content = msg["content"]
        # If content is a ChatMessageContent, extract the text
        if isinstance(content, ChatMessageContent):
            if hasattr(content, "content") and content.content:
                content = content.content
            else:
                content = str(content)
        return {"status": msg["type"], "message": content}
    return {"status": "no_new_message", "message": None}

@app.post("/renew_session",
    tags=["Session"],
    summary="Renew a session with a new authorization token",
    description="Renew a session by disconnecting MCPs and orchestration, then reconnecting with a new token. Chat history is preserved."
)
async def renew_session(
    session_id: str = Query(..., description="Session identifier to renew"),
    Authorization: str = Header(..., description="New authorization token for MCP/API access")
):
    session = chat_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Disconnect old MCPs and orchestration
    if "plugins" in session:
        await disconnect_mcps(session["plugins"])
    if "orchestration" in session:
        session["orchestration"] = None
    # Reconnect MCPs with new token
    plugins = await connect_mcps(Authorization)
    agents = await get_agents(Authorization, plugins)
    handoffs = create_handoffs(agents)
    orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        agent_response_callback=partial(agent_response_callback, session_id=session_id),
        human_response_function=make_human_response_function(session_id)
    )
    session["orchestration"] = orchestration
    session["plugins"] = plugins
    # Optionally, update created_at to extend expiry
    session["created_at"] = datetime.now(timezone.utc)
    return {"session_id": session_id, "status": "renewed"}




