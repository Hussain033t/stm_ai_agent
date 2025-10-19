import os
from dotenv import load_dotenv
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.agents import ChatCompletionAgent, OrchestrationHandoffs
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion


load_dotenv()

# ==============================
# Environment Variables
# ==============================
def require_env(var_name):
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value

API_KEY = require_env("AZURE_OPENAI_KEY")
ENDPOINT = require_env("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = require_env("AZURE_OPENAI_DEPLOYMENT")
TASK_MCP_ENDPOINT = require_env("TASK_MCP_ENDPOINT")
HELP_MCP_ENDPOINT = require_env("HELP_MCP_ENDPOINT")



async def connect_mcps(token: str):
    work_order_plugin = MCPStreamableHttpPlugin(
        name="WorkOrderMCP",
        description="Tools for managing work orders.",
        url=TASK_MCP_ENDPOINT,
    )
    labor_plugin = MCPStreamableHttpPlugin(
        name="LaborMCP",
        description="Tools for managing labor records.",
        url=TASK_MCP_ENDPOINT,
    )

    help_plugin = MCPStreamableHttpPlugin(
        name="HelpMCP",
        description="Knowledge base from help documentation.",
        url=HELP_MCP_ENDPOINT
    )

    work_order_plugin.headers = {'authorization': token}
    labor_plugin.headers = {'authorization': token}
    help_plugin.headers = {'authorization': token}

    await work_order_plugin.connect()
    await labor_plugin.connect()
    await help_plugin.connect()

    return {
        "work_order_plugin": work_order_plugin,
        "labor_plugin": labor_plugin,
        "help_plugin": help_plugin
    }

async def disconnect_mcps(plugins: dict):
    await plugins["work_order_plugin"].close()
    await plugins["labor_plugin"].close()
    await plugins["help_plugin"].close()

# ==============================
# Agents setup
# ==============================

async def get_agents(token: str, plugins: dict):
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
        plugins=[plugins["help_plugin"]]
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
        plugins=[plugins["work_order_plugin"], plugins["labor_plugin"]]
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
        plugins=[plugins["work_order_plugin"]]
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
        plugins=[plugins["labor_plugin"]]
    )

    return [main_agent, help_agent, task_agent, work_order_agent, labor_agent]

def create_handoffs(agents):
    main_agent, help_agent, task_agent, work_order_agent, labor_agent = agents
    handoffs = (
        OrchestrationHandoffs()
        .add_many(
            source_agent=main_agent.name,
            target_agents={
                help_agent.name: "Transfer to this agent if the user asks any questions related to documention or help",
                task_agent.name: "Transfer to this agent if the user requests to perform a task such as creating work order or listing labours etc..."
            }
        )
        .add_many(
            source_agent=task_agent.name,
            target_agents={
                work_order_agent.name: "Transfer to this agent if the user requests to perform a task related to workorder",
                labor_agent.name: "Transfer to this agent if the user requests to perform a task related to labour"
            }
        )
        .add(
            source_agent=work_order_agent.name,
            target_agent=task_agent.name,
            description="Transfer to this agent if the user's request is not related to performing a task on work orders"
        )
        .add(
            source_agent=work_order_agent.name,
            target_agent=task_agent.name,
            description="Transfer to this agent if the user's request is not related to performing a task on labors"
        )
        .add(
            source_agent=task_agent.name,
            target_agent=main_agent.name,
            description="Transfer to this agent if the user's request is not related to performing a task"
        )
        .add(
            source_agent=help_agent.name,
            target_agent=main_agent.name,
            description="Transfer to this agent if the user's request is not related to documentation"
        )
    )
    return handoffs

