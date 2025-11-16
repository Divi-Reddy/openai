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
    print("📡 OpenAI Key Provided:", "YES" if req.openaiKey else "NO")
    print("🏛 Decisions Base URL:", req.decisionsBaseUrl)
    print("🔑 Session ID:", req.sessionId)
    print("🧰 Functions provided:", len(req.functions))

    # Create OpenAI client with runtime key
    client = OpenAI(api_key=req.openaiKey)

    # Prepare OpenAI function schema
    openai_functions = [
        {
            "name": fn["name"],
            "description": fn["description"],
            "parameters": fn["parameters"]
        }
        for fn in req.functions
    ]

    print("\n📝 Final OpenAI Function Schema Sent:")
    print(json.dumps(openai_functions, indent=2))

    # STEP 1 — Ask OpenAI what function to call
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",               # ✅ GPT-5 NANO ENABLED
        messages=req.history + [{"role": "user", "content": req.message}],
        functions=openai_functions,
        function_call="auto"
    )

    msg = response.choices[0].message

    print("\n🤖 RAW OPENAI RESPONSE:")
    print(msg)

    # ===========================================
    # FUNCTION CALL HANDLING (new OpenAI API)
    # ===========================================
    if msg.function_call:

        fn_name = msg.function_call.name
        args = json.loads(msg.function_call.arguments)

        print("\n🛠️ OpenAI Requested Function:", fn_name)
        print("🔧 Arguments Provided:", args)

        # Match the function definition sent in request body
        selected_fn = next(f for f in req.functions if f["name"] == fn_name)

        print("\n📌 Matched Function Config:")
        print(json.dumps(selected_fn, indent=2))

        # Build Decisions API URL
        url = req.decisionsBaseUrl + selected_fn["endpoint"]
        method = selected_fn.get("httpMethod", "GET").upper()

        print("\n🌐 Calling Decisions API:")
        print("➡ URL:", url)
        print("➡ Method:", method)

        params = {"sessionid": req.sessionId}

        # GET request
        if method == "GET":
            params.update(args)
            print("➡ Query Params:", params)
            api_response = requests.get(url, params=params)

        # POST request
        else:
            print("➡ Body:", args)
            api_response = requests.post(url, params=params, json=args)

        print("\n📨 RAW Decisions API Response:")
        print(api_response.text)

        try:
            system_result = api_response.json()
        except:
            system_result = {"error": "Invalid JSON response from Decisions API"}

        print("\n📤 Sending result to OpenAI (GPT-5 Nano) for final answer...")

        final = client.chat.completions.create(
            model="gpt-5.1-nano",           # ✅ ALSO HERE
            messages=[
                {"role": "assistant", "function_call": msg.function_call},
                {"role": "tool", "name": fn_name, "content": json.dumps(system_result)}
            ]
        )

        print("\n✅ FINAL OPENAI RESPONSE:")
        print(final.choices[0].message.content)

        return {"response": final.choices[0].message.content}

    # ===========================================
    # NO FUNCTION CALL — Direct text response
    # ===========================================
    print("\n💬 DIRECT RESPONSE (No function needed):")
    print(msg.content)

    return {"response": msg.content}
