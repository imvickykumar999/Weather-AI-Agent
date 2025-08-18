import os
import re
import asyncio
import warnings
import logging
from dotenv import load_dotenv

# --- Imports from your stack ---
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # Multi-provider wrapper (Groq supported)
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  # For creating message Content/Parts
from WeatherAPI import get_weather  # Your tool

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

print("Libraries imported.")
print("\nEnvironment configured.")

# Optional: quick tool smoke test (safe to remove)
try:
    print(get_weather("New York"))
    print(get_weather("Paris"))
except Exception as e:
    print(f"[WeatherAPI test] Skipped due to: {e}")

# ----------------------------
# Model Constants (Groq-first)
# ----------------------------

GROQ_PRIMARY = "groq/llama3-70b-8192"
GROQ_FALLBACKS = [
    "groq/llama3-8b-8192",
    "groq/gemma2-9b-it",
]

# ADK/LiteLlm uses OpenAI-style params; keep outputs short to reduce TPM.
DEFAULT_MAX_TOKENS = 256

def make_groq_model(model_name: str) -> LiteLlm:
    return LiteLlm(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        # Most OpenAI-compatible providers accept "max_tokens"
        max_tokens=DEFAULT_MAX_TOKENS,
        # you can also add temperature/top_p if you want:
        # temperature=0.2,
        # top_p=0.9,
        # LiteLLM will do its own retrying if configured globally; we handle retries below anyway.
    )

# ----------------------------
# Define the Weather Agent
# ----------------------------
weather_agent = Agent(
    name="weather_agent_v1",
    model=make_groq_model(GROQ_PRIMARY),
    description="Provides weather information for specific cities.",
    instruction=(
        "You are a helpful weather assistant. "
        "When the user asks for the weather in a specific city, "
        "use the 'get_weather' tool to find the information. "
        "If the tool returns an error, inform the user politely. "
        "If the tool is successful, present the weather report clearly. "
        "Keep responses concise."
    ),
    tools=[get_weather],
)

print(f"Agent '{weather_agent.name}' created using model '{GROQ_PRIMARY}'.")

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
    agent=weather_agent,
    app_name=APP_NAME,
    session_service=session_service
)
print(f"Runner created for agent '{runner.agent.name}'.")

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
# Agent Interaction (with retry + fallback)
# ----------------------------
async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
    """
    Sends a query to the agent with robust rate-limit handling and model fallback.
    """
    print(f"\n>>> User Query: {query}")

    # Keep the prompt short to reduce token usage.
    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default
    max_attempts = 5
    attempted_models = {GROQ_PRIMARY}
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        try:
            # IMPORTANT: Trim conversation history to reduce tokens.
            # If your ADK Runner uses entire session history by default,
            # pass only the new message. (Runner handles context internally.)
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if getattr(event, "content", None) and event.content.parts:
                        final_response_text = event.content.parts[0].text
                    elif getattr(event, "actions", None) and getattr(event.actions, "escalate", None):
                        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    break

            print(f"<<< Agent Response: {final_response_text}")
            return  # success

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if _is_rate_limit_error(e):
                print(f"[RateLimit] Attempt {attempt}/{max_attempts}: {msg}")

                # 1) first: wait as suggested / backoff
                await _backoff_sleep(str(e), attempt)

                # 2) then: try fallback model if still failing on next loop
                # (swap only once per attempt to keep logic simple)
                if await _swap_model_to_fallback(runner.agent, attempted_models):
                    continue

                # If no fallback left, loop will retry with same model after backoff
                continue

            # Non-rate-limit error: raise or print friendly message and stop
            print(f"[Error] Unhandled exception: {msg}")
            break

    # If we reach here, retries exhausted
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
        # Let any pending SSL/HTTP2 writes flush before the loop shuts down
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

