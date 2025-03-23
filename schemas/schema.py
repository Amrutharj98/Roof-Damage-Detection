from pydantic import BaseModel


class ImageURL(BaseModel):
    image: str