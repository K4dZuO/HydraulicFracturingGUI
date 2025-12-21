from pydantic import BaseModel, Field, ConfigDict


class RequiredParams(BaseModel):
    k: float = Field(gt=0)
    h: float = Field(gt=0)
    phi: float = Field(gt=0, le=1)
    
    model_config = ConfigDict(extra='forbid')
