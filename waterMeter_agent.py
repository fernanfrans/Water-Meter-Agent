import time
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from WaterMeter_tool_registry import WaterMeterTools

# --- Logger ---
class ThinkingLogger(BaseCallbackHandler):

    def on_chain_start(self, serialized, inputs, **kwargs):
        # Only display once at the start of the whole agent
        if serialized.get("name") == "AgentExecutor":
            print("\n🔵 AGENT STARTED.")

    def on_llm_end(self, response, **kwargs):
        # Capture the agent's initial reasoning step (the one that never shows)
        try:
            txt = response.generations[0][0].text.strip()
            if txt:
                print(f"\n🧠 INITIAL THOUGHT: {txt}")
        except:
            pass

    def on_agent_action(self, action, **kwargs):
        thought = action.log.split("Action:")[0].replace("Thought:", "").strip()
        print(f"\n🧠 THOUGHT: {thought}")
        print(f"👉 ACTION: {action.tool}")

    def on_tool_end(self, output, **kwargs):
        text = str(output)
        print(f"👀 OBSERVATION: {text[:300]}..." if len(text) > 300 else f"👀 OBSERVATION: {text}")


    
def main():
    toolkit = WaterMeterTools()

    # --- DEFINE MULTI-ARGUMENT TOOLS ---
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
    
    # --- PROMPT: The "Strategy" for the Agent ---
    prompt = f"""
    You are a Smart Water Meter Reader.

    GOAL:
    Read the water meter in this image: {image_path}

    TOOLS YOU ARE ALLOWED TO USE:
    1. DetectMeterWindows
    2. ReadDigit

    IMPORTANT RULES:
    - You MUST NOT create or call any tool other than the two above.
    - You MUST NOT invent tools such as CombineDigits, ProcessOutput, etc.
    - After calling ReadDigit, you MUST STOP calling tools and finish the answer.

    STEP-BY-STEP INSTRUCTIONS:
    1. First call DetectMeterWindows with conf_threshold=0.5.
    2. If detection fails (wrong number of windows), call DetectMeterWindows again with conf_threshold=0.3.
    3. When DetectMeterWindows succeeds, it returns a list of file paths.
    4. Call ReadDigit using exactly that list of file paths.
    5. ReadDigit returns a list of dictionaries like:
    [{{'digit': 1, 'confidence': 0.95}}, ...]
    6. You MUST process this result **yourself** (without tools):
    - Extract each 'digit' in order
    - Combine digits into a single string (e.g., "12345")
    - Compute the reliability score as the AVERAGE of the 'confidence' values.

    FINAL OUTPUT FORMAT (NO TOOL CALLS):
    Final Answer: <the_number>
    Reliability Score: <average_confidence_rounded_to_3_decimals>
    """

    # Run the agent
    try:
        result = agent.run(prompt, callbacks=[ThinkingLogger()])
        print(f"\n✅ RESULT: {result}")
    except Exception as e:
        # Sometimes Llama3 finishes but creates a parsing error. 
        # The answer is often inside the error message.
        print(f"\n⚠️ Agent finished with parsing warning: {e}")

if __name__ == "__main__":
    main()