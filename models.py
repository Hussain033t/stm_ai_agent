from pydantic import BaseModel

class StartSessionResponse(BaseModel):
    session_id: str

class SendMessageRequest(BaseModel):
    session_id: str
    message: str

class SendMessageResponse(BaseModel):
    status: str

class GetResponseResponse(BaseModel):
    status: str
    message: str | None = None