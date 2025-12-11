import time
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from WaterMeter_tool_registry import WaterMeterTools

# --- 1. CLEANER LOGGER (Removed INITIAL THOUGHT) ---
class ThinkingLogger(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        if serialized.get("name") == "AgentExecutor":
            print("\n🔵 AGENT STARTED.")

    def on_agent_action(self, action, **kwargs):
        log_text = action.log
        thought = log_text.split("Action:")[0].replace("Thought:", "").strip()
        print(f"\n🧠 THOUGHT: {thought}")
        print(f"👉 ACTION: {action.tool}")

    def on_tool_end(self, output, **kwargs):
        text = str(output)
        # Shorten observation if it's too long
        print(f"👀 OBSERVATION: {text[:300]}..." if len(text) > 300 else f"👀 OBSERVATION: {text}")


def report_final_result(reading: str, reliability_score: float):
    return f"Final Answer: {reading}\nReliability Score: {reliability_score:.4f}"

def main():
    toolkit = WaterMeterTools()

    # --- DEFINE TOOLS ---
    tools = [
        StructuredTool.from_function(
            name="DetectMeterWindows",
            func=toolkit.tool_detect_windows,
            description="Detects windows. Requires 'image_path' and 'conf_threshold' (float between 0.1 and 1.0). Returns success/failure and file paths."
        ),
        StructuredTool.from_function(
            name="ReadDigit",
            func=toolkit.tool_digit_recognition,
            description="Reads digits from a LIST of image paths. Args: 'file_paths' (List[str]). Returns the identified digits."
        ),
        # WE ADD THIS TOOL TO FORCE THE CALCULATION
        StructuredTool.from_function(
            name="ReportResult",
            func=report_final_result,
            description="ALWAYS call this tool LAST. Takes 'reading' (string) and 'reliability_score' (float) to finish the job."
        )
    ]

    print("⏳ Connecting to Ollama (llama3)...")
    llm = ChatOllama(model="mistral:latest", temperature=0)

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    image_path = r"F:\Water-Meter-Agent-version1\sample_input.jpg"
    
    # --- UPDATED PROMPT ---
    prompt = f"""
    You are a Smart Water Meter Reader. Always show the thought, action, and observation steps.

    GOAL:
    Read the water meter in this image: {image_path}

    TOOLS:
    1. DetectMeterWindows (Finds the meter)
    2. ReadDigit (Reads the numbers)
    3. ReportResult (Submits the final answer)

    STEP-BY-STEP INSTRUCTIONS:
    1. Call DetectMeterWindows with conf_threshold=0.5.
    2. Call ReadDigit using the file paths from step 1.
    3. You will receive a list like [{{'digit': 0, 'confidence': 0.99}}, ...].
    4. MENTALLY and MANUALLY calculate:
       - The combined string (e.g. "00580")
       - The AVERAGE confidence score.
    5. Call ReportResult with these two values to finish.
    """

    # Run the agent
    try:
        # We assume the last tool call return value is the result
        result = agent.run(prompt, callbacks=[ThinkingLogger()])
        print(f"\n✅ RESULT: {result}")
    except Exception as e:
        print(f"\n⚠️ Agent finished with parsing warning: {e}")

if __name__ == "__main__":
    main()