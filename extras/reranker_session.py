import os
import re
import asyncio
import warnings
import logging
import json
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # Multi-provider wrapper (Groq supported)
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  # For creating message Content/Parts
from Bol7API import get_info  # Your tool

# --- Configuration ---
INTERACTIONS_FILE = "interactions.json"

# Function to save interactions to a JSON file
def save_interaction(query: str, response: str):
    interaction = {
        "query": query,
        "response": response
    }
    
    # Check if the file exists
    if os.path.exists(INTERACTIONS_FILE):
        with open(INTERACTIONS_FILE, "r") as file:
            data = json.load(file)
    else:
        data = []

    # Append the new interaction to the existing data
    data.append(interaction)

    # Write the updated data to the file
    with open(INTERACTIONS_FILE, "w") as file:
        json.dump(data, file, indent=4)

    print("Interaction saved!")

# Optional: catch LiteLLM-specific exceptions if available
try:
    import litellm  # type: ignore
except Exception:
    litellm = None

# ----------------------------
# Setup & Configuration
# ----------------------------
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file, e.g.\n\n"
        "GROQ_API_KEY=your_groq_api_key_here\n"
    )

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ----------------------------
# Model Constants (Groq-first)
# ----------------------------
GROQ_PRIMARY = "groq/llama-3.3-70b-versatile"
GROQ_FALLBACKS = [
    "groq/llama3-8b-8192",
    "groq/mixtral-8x7b-32768",
]

DEFAULT_MAX_TOKENS = 256

def make_groq_model(model_name: str) -> LiteLlm:
    return LiteLlm(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=DEFAULT_MAX_TOKENS,
    )

# ----------------------------
# Define the General Agent
# ----------------------------
general_agent = Agent(
    name="general_agent_v1",
    model=make_groq_model(GROQ_PRIMARY),
    description="Handles requests by querying external APIs to fetch and present relevant information.",
    instruction=(
        "You are a helpful assistant. "
        "When the user asks for specific information, "
        "use the available tool(s) to query the appropriate API. "
        "If the tool returns an error, inform the user politely. "
        "If the tool is successful, present the information clearly and concisely. "
        "Keep responses user-friendly and adaptable, regardless of whether the data is about weather, a company, a game, or anything else."
    ),
    tools=[get_info],  # Now using get_info as a tool
)

# ----------------------------
# Session Service & Runner
# ----------------------------
session_service = InMemorySessionService()

APP_NAME = "weather_tutorial_app"
USER_ID = "user_1"
SESSION_ID = "session_001"  # fixed for simplicity

async def create_session():
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")
    return session

runner = Runner(
    agent=general_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# ----------------------------
# Helpers: rate limit handling & model fallback
# ----------------------------
RATE_LIMIT_PATTERN = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

async def _backoff_sleep(err_msg: str, attempt: int):
    """
    Sleep using server-suggested delay if present, else exponential backoff.
    """
    m = RATE_LIMIT_PATTERN.search(err_msg or "")
    if m:
        delay = float(m.group(1))
    else:
        delay = min(2 ** attempt, 20)  # cap to avoid long stalls
    await asyncio.sleep(delay)

def _is_rate_limit_error(e: Exception) -> bool:
    if litellm and isinstance(e, getattr(litellm, "RateLimitError", tuple())):
        return True
    msg = f"{type(e).__name__}: {e}"
    return "rate limit" in msg.lower() or "rate_limit_exceeded" in msg.lower()

async def _swap_model_to_fallback(agent: Agent, used: set) -> bool:
    """
    Switch agent.model to the next fallback model not yet used.
    Returns True if swapped, False if none left.
    """
    for cand in GROQ_FALLBACKS:
        if cand not in used:
            agent.model = make_groq_model(cand)
            print(f"[Model Fallback] Switched to: {cand}")
            used.add(cand)
            return True
    return False

# ----------------------------
# Reranker Function
# ----------------------------
def rerank_responses(responses):
    """
    Ranks responses based on content specificity and structure.
    Returns the best response.
    """
    print("\n>>> Reranking responses...")

    if not responses:
        return "No valid response available."

    # Check if all responses are identical
    if len(set(responses)) == 1:
        print("All responses are identical. Returning the first response.")
        return responses[0]  # No reranking needed, return the first response

    # Initialize the best response and its score
    best_response = responses[0]
    best_score = 0

    # Score responses based on the number of bullet points or structured content
    for idx, response in enumerate(responses):
        print(f"Response {idx + 1}: {response}")

        # Simple scoring mechanism: count the number of bullet points or items in the response
        score = response.count('*')  # Count bullet points as a measure of detail

        # Add score for length or more specific content
        score += len(response.split())  # Length of the response as an additional factor

        # If the current response has more score (more detail), select it
        if score > best_score:
            best_response = response
            best_score = score

    print(f"Best Response: {best_response}")
    return best_response

# ----------------------------
# Agent Interaction (with retry + fallback)
# ----------------------------
async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
    """
    Sends a query to the agent with robust rate-limit handling and model fallback.
    """
    print(f"\n>>> User Query: {query}")

    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default
    max_attempts = 5
    attempted_models = {GROQ_PRIMARY}
    attempt = 0

    candidate_responses = []  # To store multiple candidate responses from the agent

    while attempt < max_attempts:
        attempt += 1
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if getattr(event, "content", None) and event.content.parts:
                        candidate_responses.append(event.content.parts[0].text)
                    elif getattr(event, "actions", None) and getattr(event.actions, "escalate", None):
                        candidate_responses.append(f"Agent escalated: {event.error_message or 'No specific message.'}")
                    break

            # Apply reranking to the candidate responses
            final_response_text = rerank_responses(candidate_responses)

            print(f"<<< Agent Response (after reranking): {final_response_text}")

            # Save the interaction as usual
            save_interaction(query, final_response_text)

            return  # success

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if _is_rate_limit_error(e):
                print(f"[RateLimit] Attempt {attempt}/{max_attempts}: {msg}")

                # 1) first: wait as suggested / backoff
                await _backoff_sleep(str(e), attempt)

                # 2) then: try fallback model if still failing on next loop
                if await _swap_model_to_fallback(runner.agent, attempted_models):
                    continue

                # If no fallback left, loop will retry with same model after backoff
                continue

            # Non-rate-limit error: raise or print friendly message and stop
            print(f"[Error] Unhandled exception: {msg}")
            break

    print("<<< Agent Response: We hit provider limits repeatedly. Please try again shortly or upgrade your Groq tier.")

# ----------------------------
# CLI Conversation Loop
# ----------------------------
async def run_conversation():
    await create_session()
    try:
        while True:
            try:
                user_input = input("\n>>> Enter your query (or 'exit' to quit): ")
            except (EOFError, KeyboardInterrupt):
                print("\nEnding the conversation.")
                break

            if user_input.strip().lower() == "exit":
                print("Ending the conversation.")
                break

            await call_agent_async(
                user_input.strip(),
                runner=runner,
                user_id=USER_ID,
                session_id=SESSION_ID
            )
    finally:
        await asyncio.sleep(0.25)

# ----------------------------
# Main
# ----------------------------
async def main():
    try:
        await run_conversation()
    finally:
        # Cancel any lingering tasks cleanly
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # Final small delay for sockets to drain
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
