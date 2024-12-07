
from fastapi import FastAPI, Request,HTTPException
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware
import openai
# 允许具体的源
origins = [
    "http://localhost:8080",  # 允许的源
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 指定允许的源
    allow_credentials=True,  # 允许携带凭据
    allow_methods=["*"],     # 允许的HTTP方法
    allow_headers=["*"],     # 允许的HTTP头
)

@app.get("/syh/{desc}")
async def CriticalUserPaths(desc: str):

    prompt = desc
    openai.api_base = "https://api.wlai.vip/v1"
    openai.api_key = "sk-7myVv6BVkXP7Ryit99Ce60719cFb4154A8A5B0181018684a"
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
        {"role": "user", "content": prompt}
        ],
        temperature = 0,
        max_tokens = 2056,
        )
    response = completion.choices[0].message.content
    
    print(response)

    return response  
