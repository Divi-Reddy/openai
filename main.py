
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

    client = OpenAI(api_key=req.openaiKey)

    openai_functions = [
        {
            "name": fn["name"],
            "description": fn["description"],
            "parameters": fn["parameters"]
        }
        for fn in req.functions
    ]

    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=req.history + [{"role": "user", "content": req.message}],
        functions=openai_functions,
        function_call="auto"
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        call = msg.tool_calls[0]
        fn_name = call.name
        args = json.loads(call.arguments)

        selected_fn = next(f for f in req.functions if f["name"] == fn_name)

        url = req.decisionsBaseUrl + selected_fn["endpoint"]
        method = selected_fn.get("httpMethod", "GET").upper()

        params = {"sessionid": req.sessionId}

        if method == "GET":
            params.update(args)
            api_response = requests.get(url, params=params)
        else:
            api_response = requests.post(url, params=params, json=args)

        system_result = api_response.json()

        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "assistant", "tool_calls": msg.tool_calls},
                {"role": "tool", "name": fn_name, "content": json.dumps(system_result)}
            ]
        )

        return {"response": final.choices[0].message["content"]}

    return {"response": msg["content"]}
