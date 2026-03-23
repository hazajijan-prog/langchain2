import asyncio
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream_async
from util.pretty_print import print_mcp_tools, get_user_input

# Middleware: Hantera tool-output
@wrap_tool_call
async def handle_tool_output(request, handler):
    # Kör tool-anropet
    result = await handler(request)

    # Returnerar resultatet direkt utan modifiering
    # (uppfyller kravet att output passerar middleware)
    return result

# Huvudfunktion (async)
async def run_async():
    
    # Skapa MCP-klient och koppla till vår server
    mcp_client = MultiServerMCPClient({
        "planner_server": {
            "transport": "streamable_http",
            "url": "http://localhost:8003/mcp",
        }
    })
    # Hämta alla tools från MCP-servern
    tools = await mcp_client.get_tools()
    
    # Filtrera vilka tools agenten får använda
    allowed_tools = [
        "extract_tasks",
        "extract_times",
        "extract_durations",
        "create_schedule",
        "format_schedule",
    ]
    # Behåll endast tillåtna tools
    tools = [tool for tool in tools if tool.name in allowed_tools]
    
    # Skriv ut tools (för debugging)
    print_mcp_tools(tools)
    
    # Koppla modellen till verktygen
    model = get_model().bind_tools(tools)
    
    # Skapa agent
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[handle_tool_output],         # Middleware som körs vid tool calls
        system_prompt=("""
            <role>
                Du är en personlig planeringsassistent.
                Du hjälper användaren att strukturera sin dag.
                Svara alltid på svenska.
                </role>

                <tools>
                Du har tillgång till dessa verktyg:
                - extract_tasks
                - extract_times
                - extract_durations
                - create_schedule
                - format_schedule
                </tools>

                <workflow>
                Du MÅSTE alltid följa exakt denna ordning:

                1. extract_tasks
                2. extract_times
                3. extract_durations
                4. create_schedule
                5. format_schedule

                Du får aldrig hoppa över ett steg.
                Du får aldrig ändra ordningen.
                Du får aldrig gå direkt till create_schedule.
                </workflow>

                <rules>
                Du får aldrig skriva vanlig text.

                Du får inte förklara vad du gör.
                Du får inte beskriva nästa steg.
                Du ska direkt anropa nästa verktyg.

                Du måste börja med extract_tasks.

                När du anropar extract_tasks:
                - du måste alltid skicka originaltexten från användaren
                - du får inte omformulera texten
                - du får inte korta ner texten
                - argumentet "text" måste vara exakt användarens ursprungliga text

                När du anropar extract_times:
                - du måste alltid skicka argumentet "text" som en sträng
                - använd originaltexten från användaren

                När du anropar extract_durations:
                - skicka text som sträng
                - använd originaltexten från användaren

                När du anropar create_schedule:
                - använd resultatet från extract_tasks som "tasks"
                - använd resultatet från extract_times som "times"

                Efter create_schedule:
                - du måste direkt anropa format_schedule

                När du anropar format_schedule:
                - skicka resultatet från create_schedule som "schedule"

               SLUTREGLER:

                - Ditt sista svar ska vara exakt i detta format:

                Din dag har strukturats på följande sätt:

                1. ...
                2. ...
                3. ...

                Spara detta schema för att planera din dag!

                - Använd numrerad lista (1, 2, 3)
                - Använd exakt formulering
                - Ingen extra text före eller efter

                FEL:
                - Om du skriver text → fel
                - Om du hoppar över extract_times → fel
                - Om du inte använder format_schedule → fel
                </rules>
        """),
    )
    # Hämta input från användaren
    user_input = get_user_input("Beskriv din dag")
    # Skicka input till agenten (streaming)
    process_stream =  agent.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode=STREAM_MODES,
        )
    # Hantera och visa output
    await handle_stream_async(process_stream)
    
# Startfunktion
def run():
    asyncio.run(run_async())

# Kör programmet
if __name__ == "__main__":
    run()
    