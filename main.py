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

    client = OpenAI(api_key=req.openaiKey)

    # Prepare OpenAI function schema (only name/description/parameters)
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

    # Step 1 → Ask OpenAI what to do
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=req.history + [{"role": "user", "content": req.message}],
        functions=openai_functions,
        function_call="auto"
    )

    msg = response.choices[0].message
    print("\n🤖 RAW OPENAI RESPONSE:")
    print(msg)

    # If OpenAI requests a function call
    if msg.tool_calls:
        call = msg.tool_calls[0]
        fn_name = call.name
        args = json.loads(call.arguments)

        print("\n🛠️ OpenAI Requested Function:", fn_name)
        print("🔧 Arguments Provided:", args)

        # Find matching function definition
        selected_fn = next(f for f in req.functions if f["name"] == fn_name)

        print("\n📌 Matched Function Config:")
        print(json.dumps(selected_fn, indent=2))

        # Build full API URL
        url = req.decisionsBaseUrl + selected_fn["endpoint"]
        method = selected_fn.get("httpMethod", "GET").upper()

        print("\n🌐 Calling Decisions API:")
        print("➡ URL:", url)
        print("➡ Method:", method)

        params = {"sessionid": req.sessionId}

        # Execute GET / POST dynamically
        if method == "GET":
            params.update(args)
            print("➡ Query Params:", params)
            api_response = requests.get(url, params=params)
        else:
            print("➡ Body:", args)
            api_response = requests.post(url, params=params, json=args)

        print("\n📨 RAW Decisions API Response:")
        print(api_response.text)

        try:
            system_result = api_response.json()
        except:
            print("❌ ERROR: Decisions API did not return valid JSON!")
            system_result = {"error": "Invalid JSON response from Decisions API"}

        # Step 2 → Send result back to OpenAI to summarize
        print("\n📤 Sending result to OpenAI for final answer...")
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "assistant", "tool_calls": msg.tool_calls},
                {"role": "tool", "name": fn_name, "content": json.dumps(system_result)}
            ]
        )

        print("\n✅ FINAL OPENAI RESPONSE:")
        print(final.choices[0].message["content"])

        return {"response": final.choices[0].message["content"]}

    # No tool call → take direct response
    print("\n💬 DIRECT RESPONSE (No function needed):")
    print(msg["content"])

    return {"response": msg["content"]}
