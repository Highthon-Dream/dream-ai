from pydantic.v1 import BaseModel
import datetime

# === CONFIGS ====

MODEL_INFOS = {"mistral": "/Users/ysh/Desktop/fun/bucketlist_ai/model_store/mistral-7b-instruct-v0.2.Q2_K.gguf", "vicuna": "/Users/ysh/Desktop/fun/bucketlist_ai/model_store/vicuna-13b-v1.5.Q2_K.gguf"}


# === USER ====

class UserItem(BaseModel):
    user_id: str
    bucketlists: list[str]
    questions: list[str]

class UserOject(BaseModel):
    created_at: datetime.datetime
    id: int
    identify: str
    name: str

# === BUCKET LIST ====

class BucketGoalTemplate(BaseModel):
    name: str
    desc: str

class BucketListObject(BaseModel):
    created_at: datetime.datetime
    id: int
    user_id: int
    content: str
    title: str

# ==== Question ====
    
class QuestinoTemplate(BaseModel):
    created_at: datetime.datetime
    id: int
    user_id: int
    content: str
    title: str