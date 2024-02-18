from typing import Union, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from app.db.client import make_client
from app.nlp.llamacpp_hook import LLMAgent
from app.recommend import fetch_users, fetch_bucketlist, fetch_questions, RecommendationSystem

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Make Client
host = "192.168.10.231:3306"
username = 'root'
password = '1234'
database = 'weego'
client = make_client(host, username, password, database)

agent = LLMAgent("koalpaca")
recommendation_system = RecommendationSystem(client=client)

class BucketListReq(BaseModel):
    goal: str
    num_items: int=5
    search: bool=False
    streaming: bool=False

@app.post("/recommendation/bucketlist")
async def specify_bucketlist(req: BucketListReq):
    """
        About 10~16s for full response.
    """
    if not req.streaming:
        bucketlist_recommendation = [i for i in agent.get_answer(req.goal, search=req.search, num_items=req.num_items, streaming=False)][0]
        return bucketlist_recommendation
    
    return StreamingResponse(agent.get_answer(req.goal, search=req.search, num_items=req.num_items, streaming=True), media_type="application/x-ndjson")


# @app.put("/recommendation")
# def update_recommendation():
QUESTION = "question"
BUCKETLIST = "bucketlist"

class RecommendationReq(BaseModel):
    user_id: int
    using: str

@app.post("/recommendation/user")
def get_user_recommendation(req: RecommendationReq):
    if req.using not in [QUESTION, BUCKETLIST]: return
    if req.using == QUESTION:
        data = fetch_questions(client=client)
    elif req.using == BUCKETLIST:
        data = fetch_bucketlist(client=client)

    users = recommendation_system.recommend_user(req.user_id, data)

    return users