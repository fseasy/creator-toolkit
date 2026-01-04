from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class FooInput(BaseModel):
    text: str
    repeat: int = 1


class FooOutput(BaseModel):
    result: str


@router.post("/tts", response_model=FooOutput)
def tts(input: FooInput):
    return FooOutput(result=input.text * input.repeat)
