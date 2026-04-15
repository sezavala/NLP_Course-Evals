from typing import Literal, List, Optional
from pydantic import BaseModel, Field

class SentimentScore(BaseModel):
    rubric: Literal["AI_MODEL"] = Field(
        default="AI_MODEL",
        description="Scoring rubric/source used for sentiment."
    )
    score: int = Field(
        ge=-3,
        le=3,
        description="Sentiment intensity: -3 (strong negative) to +3 (strong positive), 0 for N/A."
    )
    intensity: Literal["Weak", "Medium", "Strong"] = Field(
        description="Intensity/force of the sentiment."
    )
    label: Literal["negative", "neutral", "positive"] = Field(
        description="Discrete sentiment bucket."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the sentiment classification."
    )

class Comment(BaseModel):
    text: str = Field(description="The original feedback comment.")
    sentiment: Optional[SentimentScore] = Field(
        default=None,
        description="The sentiment score of the comment, if applicable."
    )

class TopicCategorization(BaseModel):
    topic: str = Field(description="The topic of the input text.")
    feedback: List[Comment]

class AllTopics(BaseModel):
    topics: List[TopicCategorization]