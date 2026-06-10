from typing import Literal

from pydantic import BaseModel, Field


AlgorithmType = Literal[
    "regression",
    "classification",
    "clustering",
    "nlp",
    "neural_network",
]


class PipelineNode(BaseModel):
    id: str
    type: AlgorithmType
    label: str


class PipelineEdge(BaseModel):
    source: str
    target: str


class PipelineRequest(BaseModel):
    algorithm_type: AlgorithmType
    nodes: list[PipelineNode] = Field(default_factory=list)
    edges: list[PipelineEdge] = Field(default_factory=list)
    sample_size: int = Field(default=300, ge=50, le=5000)


class PipelineResult(BaseModel):
    algorithm_type: AlgorithmType
    metric_name: str
    metric_value: float
    setup_time_manual_min: float
    setup_time_builder_min: float
    setup_time_reduction_pct: float
    explanation: str
