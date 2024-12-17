from pydantic import BaseModel


class Inputs(BaseModel):
    address: str
    name: str
    rating: str
    rubrics: str
