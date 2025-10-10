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
        url="https://stm-ai-task-mcp-bjf8d2d3fraphude.eastus2-01.azurewebsites.net/mcp",
    )
labor_plugin = MCPStreamableHttpPlugin(
        name="LaborMCP",
        description="Tools for managing labor records.",
        url="https://stm-ai-task-mcp-bjf8d2d3fraphude.eastus2-01.azurewebsites.net/mcp",
    )

help_plugin = MCPStreamableHttpPlugin(
    name="HelpMCP",
    description="Knowledge base from help documentation.",
    url="https://stm-ai-help-mcp-h2fga6hmgyascygk.eastus2-01.azurewebsites.net/mcp",
)


# ==============================
# Agents setup
# ==============================
async def get_agents(token: str):

    work_order_plugin.headers = {'authorization' : token}
    labor_plugin.headers = {'authorization' : token}
    
    await work_order_plugin.connect()
   
    await labor_plugin.connect()

    await help_plugin.connect()

     # --- Main agent ---
    main_agent = ChatCompletionAgent(
        service=AzureChatCompletion(
            deployment_name=DEPLOYMENT_NAME, endpoint=ENDPOINT, api_key=API_KEY
        ),
        name="MainAgent",
        instructions=(
            "You are the entry point agent. Route requests as follows:\n"
            "- Requests about work orders or labor → forward to TaskAgent.\n"
            "- Requests about help/documentation → forward to HelpAgent."
        )
    )

    # --- Help agent ---
    help_agent = ChatCompletionAgent(
        service=AzureChatCompletion(
            deployment_name=DEPLOYMENT_NAME, endpoint=ENDPOINT, api_key=API_KEY
        ),
        name="HelpAgent",
        instructions=("You are the **Help and Documentation Agent**.\n"
        " Your role is to assist users by providing help and documentation based on the available resources.\n"
        " Use the provided tool response to fetch relevant information and answer user queries effectively.\n"
        " Always ensure your responses are clear, concise, and directly address the user's questions.\n"
        " If you cannot find the information, politely inform the user that the documentation does not cover their request.\n"
        " 📋 **Response Formatting Rules:**\n"
        " - Provide answers in a clear and structured manner.\n"
        " - Use bullet points or numbered lists for clarity when appropriate.\n"
        " - Convert the response from the tool into a user-friendly format based on the question.\n"
        " - If the tool response is empty or not helpful, inform the user that the documentation does not cover their request.\n"
        " - Show the URL from the tool response if applicable at the end."
        ),
        plugins=[help_plugin]
    )

    task_agent = ChatCompletionAgent(
        service=AzureChatCompletion(
            deployment_name=DEPLOYMENT_NAME, endpoint=ENDPOINT, api_key=API_KEY
        ),
        name="TaskAgentOrchestration",
        instructions=(
            "You are the **Task Orchestrator Agent**.\n"
            "Your role is to coordinate between the WorkOrderAgent and the LaborAgent.\n\n"
            "🔹 Routing Rules:\n"
            "- If the user’s request is only about **work orders** (create, update, delete, list, check status), "
            "forward it to WorkOrderAgent.\n"
            "- If the request is only about **labor** (create labor entry, update labor, list technicians, delete labor), "
            "forward it to LaborAgent.\n"
            "🔹 Response Guidelines:\n"
            "- Clearly confirm each operation result back to the user.\n"
            "- Use JSON when confirming API actions, with `status` and `details` fields.\n"
            "  Example: {\"status\": \"success\", \"details\": \"Labor LAB-5 added\"}\n\n"
            "You never answer user requests directly — you always delegate to one of the appropriate specialist agent."
        ),
        plugins=[work_order_plugin, labor_plugin]
    )

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

    return [main_agent, help_agent, task_agent, work_order_agent, labor_agent]

def create_handoffs(agents):
    main_agent, help_agent, task_agent, work_order_agent, labor_agent = agents
    handoffs = (
        OrchestrationHandoffs()
        .add(
            source_agent=main_agent.name,
            target_agent=help_agent.name,
            description="If user asks any questions related to help or documentation"
        )
        .add(
            source_agent=main_agent.name,
            target_agent=task_agent.name,
            description="If user asks about work orders or labor"
        )
        .add(
            source_agent=task_agent.name,
            target_agent=work_order_agent.name,
            description="If user asks to create, update, or delete work orders"
        )
        .add(
            source_agent=task_agent.name,
            target_agent=labor_agent.name,
            description="If user asks to create, update, or delete labor or technicians"
        )
    )
    return handoffs 
