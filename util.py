import asyncio
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent




# ==============================
# Chat sessions (per user)
# ==============================
chat_sessions: dict[str, dict] = {}



# ==============================
# Agent response callback
# ==============================

def agent_response_callback(message: ChatMessageContent, session_id: str):
    # Print the main message only once
    print(message)
    if message.content.strip():
        print(f"{message.name}: {message.content}")
        print("from agent_response_callback")
        session = chat_sessions.get(session_id)
        if session is not None:
            session["last_agent_question"] = message.content
    # Print only tool calls/results
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"🔧 Calling function '{item.name}' with arguments: {item.arguments}")
        elif isinstance(item, FunctionResultContent):
            print(f"✅ Result from '{item.name}': {item.result}")

   




def make_human_response_function(session_id: str):
    async def _human_response():
        session = chat_sessions[session_id]
        question = session.get("last_agent_question", "Agent needs input.")
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
    thread = session["thread"]
    orchestration = session["orchestration"]
    runtime = session["runtime"]
    # Add user message to thread
    thread.append(ChatMessageContent(role=AuthorRole.USER, content=user_message))
    # Prepare recent history for prompt
    recent_msgs = thread[-10:] if len(thread) > 10 else thread
    history_prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in recent_msgs)
    task_prompt = f'Conversation so Far:{history_prompt} Current Request: {user_message}'
    orchestration_result = await orchestration.invoke(
        task=task_prompt,
        runtime=runtime
    )
    value = await orchestration_result.get()
    # Save agent reply to thread and queue for polling
    thread.append(value)
    session["unpolled_agent_messages"].append({
        "role": "assistant",
        "content": value,
        "type": "agent_response"
    })