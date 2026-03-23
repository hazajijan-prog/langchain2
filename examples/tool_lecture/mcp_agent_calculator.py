import asyncio
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream_async
from util.pretty_print import print_mcp_tools, get_user_input


#  Middleware: fångar output från verktyg innan det når agenten
@wrap_tool_call
async def handle_tool_output(request, handler):
    # Kör själva tool-anropet (t.ex add_numbers)
    result = await handler(request)

    # Om resultatet är ett ToolMessage → modifiera det
    if isinstance(result, ToolMessage):
        return ToolMessage(
            content=f"Resultat från verktyg: {result.content[0]['text']}",
            tool_call_id=result.tool_call_id,
        )
        
    # Om något annat → skicka vidare som det är
    return result


async def run_async():
    # Hämtar LLM-modellen
    model = get_model()

    # Skapar MCP-klient som kopplar till vår server
    mcp_client = MultiServerMCPClient({
        "math_server": {
            "transport": "streamable_http",
            "url": "http://localhost:8001/mcp",
        }
    })

    # Hämtar ALLA tools från MCP-servern

    tools = await mcp_client.get_tools()

    # Filtrering
    # Agenten ska bara få tillgång till vissa tools
    allowed_tools = ["add_numbers", "subtract_numbers", "multiply_numbers"]
    
    # Behåller bara tools som finns i allowed_tools
    tools = [tool for tool in tools if tool.name in allowed_tools]

    # Skriver ut vilka tools agenten faktiskt har tillgång till
    print_mcp_tools(tools)

    #   Skapar agenten och kopplar:
    # - modellen
    # - tools (filtrerade)
    # - middleware (för att ändra output)
    
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[handle_tool_output],  
        system_prompt=("""

                        <role>
                        Du är en matematisk assistent som använder tillgängliga verktyg från MCP-servern för att utföra beräkningar.
                        </role>

                        <workflow>
                        När användaren ställer en matematisk fråga:
                        1. Identifiera vilken typ av beräkning som efterfrågas.
                        2. Välj rätt verktyg bland de verktyg som är tillgängliga för agenten.
                        3. Anropa endast ett verktyg i taget.
                        4. Vänta på resultatet från verktyget innan du formulerar ditt svar.
                        5. Svara utifrån resultatet från verktyget.

                        Om ingen lämplig tool finns tillgänglig för frågan:
                        Säg att rätt verktyg inte finns tillgängligt för agenten.
                        </workflow>

                        <rules>
                        Du får inte utföra matematiska beräkningar själv.
                        Du måste använda ett tillgängligt verktyg för alla matematiska uträkningar.
                        Du får inte hitta på resultat.
                        Du får endast använda de verktyg som agenten har tillgång till.
                        Om ett verktyg returnerar ett felmeddelande ska du återge detta tydligt och kortfattat på svenska.
                        Svara alltid på svenska.
                        Håll svaret kort, tydligt och sakligt.
                        </rules>

                        <output_format>
                        Vanliga svar ska vara korta och tydliga.
                        Om en beräkning lyckas ska svaret presenteras som ett kort resultat på svenska.
                        Om ett verktyg inte finns tillgängligt ska svaret tydligt säga detta.
                        Om ett verktyg returnerar fel ska svaret återge felet kortfattat.
                        </output_format>

                        <examples>
                        Exempel:
                        Fråga: Vad är 2 plus 3?
                        Använd verktyg: add_numbers
                        Svar: Resultatet är 5.

                        Fråga: Vad är 10 minus 4?
                        Använd verktyg: subtract_numbers
                        Svar: Resultatet är 6.

                        Fråga: Vad är roten ur 9?
                        Om verktyget inte finns tillgängligt för agenten:
                        Svar: Jag har inget tillgängligt verktyg för den beräkningen.
                        </examples
                        """

        ),
    )

    # Hämtar input från användaren
    user_input = get_user_input("Ställ din fråga")

    # Skickar frågan till agenten
    process_stream = agent.astream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=STREAM_MODES,
    )

    # Hanterar streaming output (oförändrad från läraren)
    await handle_stream_async(process_stream)


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()