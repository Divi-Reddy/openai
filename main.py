from fastapi import FastAPI
from pydantic import BaseModel
import json, requests
from openai import OpenAI

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    openaiKey: str
    decisionsBaseUrl: str
    sessionId: str
    functions: list
    history: list = []


@app.post("/chat")
def chat(req: ChatRequest):

    print("\n==============================")
    print("🔥 NEW REQUEST RECEIVED")
    print("==============================")
    print("🧠 Prompt:", req.message)
    print("📡 Using Model: gpt-5-nano-2025-08-07")

    client = OpenAI(api_key=req.openaiKey)

    # Build NEW OpenAI tool schema format
    tools = []
    for f in req.functions:
        tools.append({
            "type": "function",
            "function": {
                "name": f["name"],
                "description": f["description"],
                "parameters": f["parameters"]
            }
        })

    print("\n🧰 TOOLS SENT TO OPENAI:")
    print(json.dumps(tools, indent=2))

    # STEP 1: Ask GPT what tool to call
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=[{"role": "user", "content": req.message}],
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    print("\n🤖 RAW MODEL RESPONSE:")
    print(msg)

    # ============================================================
    # FUNCTION CALL HANDLING (NEW TOOL-CALL PROTOCOL)
    # ============================================================
    if msg.tool_calls:

        tool_call = msg.tool_calls[0]
        tool_call_id = tool_call.id
        fn_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print("\n🛠️ TOOL REQUESTED:", fn_name)
        print("🔧 Arguments:", args)
        print("🆔 tool_call_id:", tool_call_id)

        # Match your dynamic function
        selected_fn = next(f for f in req.functions if f["name"] == fn_name)

        print("\n📌 MATCHED FUNCTION CONFIG:")
        print(json.dumps(selected_fn, indent=2))

        # Build Decisions URL
        url = req.decisionsBaseUrl + selected_fn["endpoint"]
        method = selected_fn.get("httpMethod", "GET").upper()

        print("\n🌐 Calling Decisions API:")
        print("➡ URL:", url)
        print("➡ Method:", method)

        params = {"sessionid": req.sessionId}

        if method == "GET":
            params.update(args)
            api_response = requests.get(url, params=params)
        else:
            api_response = requests.post(url, params=params, json=args)

        print("\n📨 RAW DECISIONS RESPONSE:")
        print(api_response.text)

        try:
            tool_result = api_response.json()
        except:
            tool_result = {"error": "Invalid JSON returned by Decisions"}

        print("\n📤 Sending tool result back to GPT for final answer...")

        # ============================================================
        # STEP 2: Return tool result using OpenAI’s NEW MESSAGE FORMAT
        # ============================================================
        final = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {"role": "user", "content": req.message},

                # Assistant message containing tool_calls (REQUIRED)
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": json.dumps(args)
                            }
                        }
                    ]
                },

                # Tool response (REQUIRED matching ID)
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result)
                }
            ]
        )

        final_answer = final.choices[0].message.content

        print("\n✅ FINAL GPT RESPONSE:")
        print(final_answer)

        return {"response": final_answer}

    # ============================================================
    # NO TOOL CALL → Direct model answer
    # ============================================================
    print("\n💬 DIRECT RESPONSE (No tools needed):")
    print(msg.content)

    return {"response": msg.content}
