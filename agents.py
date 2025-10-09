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

load_dotenv()

# ==============================
# Environment Variables
# ==============================
API_KEY = os.getenv("AZURE_OPENAI_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")


work_order_plugin = MCPStreamableHttpPlugin(
        name="WorkOrderMCP",
        description="Tools for managing work orders.",
        url="http://0.0.0.0:8080/mcp",
    )
labor_plugin = MCPStreamableHttpPlugin(
        name="LaborMCP",
        description="Tools for managing labor records.",
        url="http://0.0.0.0:8080/mcp",
    )


# ==============================
# Agents setup
# ==============================
async def get_agents(token: str):

    work_order_plugin.headers = {'authorization' : token}
    labor_plugin.headers = {'authorization' : token}
    
    await work_order_plugin.connect()
   
    await labor_plugin.connect()

    work_order_agent = ChatCompletionAgent(
        service=AzureChatCompletion(
            deployment_name=DEPLOYMENT_NAME,
            endpoint=ENDPOINT,
            api_key=API_KEY
        ),
        name="WorkOrderAgent",
        instructions=(
            "You are the **Work Order Specialist**.\n"
            "- You manage work orders only.\n"
            "- You can create, delete, and list work orders using the provided tools.\n"
            "- Always confirm the operation result back to the user clearly.\n"
            "- If a user’s request involves labor or technicians, hand off to the LaborAgent.\n\n"
            "📋 **Response Formatting Rules:**\n"
            "- When listing work orders, always return in readable format.\n"
            "- When confirming creation or deletion, return JSON with `status` and `details` fields.\n"
            "  Example:\n"
            "  {\"status\": \"success\", \"details\": \"Work order WO-3 created for Site-C (Repair)\"}"
        ),
        plugins=[work_order_plugin]
    )

    labor_agent = ChatCompletionAgent(
        service=AzureChatCompletion(
            deployment_name=DEPLOYMENT_NAME,
            endpoint=ENDPOINT,
            api_key=API_KEY
        ),
        name="LaborAgent",
        instructions=(
            "You are the **Labor Specialist**.\n"
            "- You manage labor records only.\n"
            "- You can create, delete, and list labor entries using the provided tools.\n"
            "- Always confirm the operation result back to the user clearly.\n"
            "- If a user’s request involves work orders, hand off to the WorkOrderAgent.\n\n"
            "📋 **Response Formatting Rules:**\n"
            "- When listing labors, always return them in readable format.\n"
            "- When confirming creation or deletion, return JSON with `status` and `details` fields.\n"
            "  Example:\n"
            "  {\"status\": \"success\", \"details\": \"Labor LAB-3 created for technician 'Charlie'\"}"
        ),
        plugins=[labor_plugin]
    )

    return [work_order_agent, labor_agent]

def create_handoffs(agents):
    work_order_agent, labor_agent = agents
    handoffs = (
        OrchestrationHandoffs()
        .add(
            source_agent=work_order_agent.name,
            target_agent=labor_agent.name,
            description="If user only asks about labor or technicians"
        )
        .add(
            source_agent=labor_agent.name,
            target_agent=work_order_agent.name,
            description="If user asks about work orders"
        )
    )
    return handoffs 