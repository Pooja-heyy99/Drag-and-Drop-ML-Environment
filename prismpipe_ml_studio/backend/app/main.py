from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline_engine import execute_pipeline
from app.schemas import PipelineRequest


app = FastAPI(title="PrismPipe ML Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALGORITHMS = [
    {
        "type": "regression",
        "title": "Regression",
        "description": "Predict continuous values such as prices or demand.",
    },
    {
        "type": "classification",
        "title": "Classification",
        "description": "Predict categories such as churn risk or label class.",
    },
    {
        "type": "clustering",
        "title": "Clustering",
        "description": "Group similar records with unsupervised learning.",
    },
    {
        "type": "nlp",
        "title": "NLP",
        "description": "Build text workflows such as sentiment analysis.",
    },
    {
        "type": "neural_network",
        "title": "Neural Network",
        "description": "Train a compact neural network for non-linear patterns.",
    },
]

TUTORIALS = {
    "regression": [
        "Drop Regression node into canvas.",
        "Set sample size and feature assumptions.",
        "Click Run to view RMSE and setup-time savings.",
    ],
    "classification": [
        "Drop Classification node into canvas.",
        "Configure class target and train/test split.",
        "Run and inspect Accuracy output.",
    ],
    "clustering": [
        "Drop Clustering node into canvas.",
        "Choose cluster count and feature space.",
        "Run and inspect silhouette score.",
    ],
    "nlp": [
        "Drop NLP node into canvas.",
        "Choose text preprocessing options.",
        "Run to inspect sentiment accuracy.",
    ],
    "neural_network": [
        "Drop Neural Network node into canvas.",
        "Select hidden-layer layout.",
        "Run and inspect NN accuracy and setup-time gain.",
    ],
}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/algorithms")
def get_algorithms() -> list[dict]:
    return ALGORITHMS


@app.get("/api/tutorials/{algorithm_type}")
def get_tutorial_steps(algorithm_type: str) -> dict:
    if algorithm_type not in TUTORIALS:
        raise HTTPException(status_code=404, detail="Unsupported algorithm type")
    return {"algorithm_type": algorithm_type, "steps": TUTORIALS[algorithm_type]}


@app.post("/api/pipeline/execute")
def run_pipeline(payload: PipelineRequest) -> dict:
    if not payload.nodes:
        raise HTTPException(status_code=400, detail="At least one pipeline node is required")

    node_types = {node.type for node in payload.nodes}
    if payload.algorithm_type not in node_types:
        raise HTTPException(
            status_code=400,
            detail="Selected algorithm must exist in dropped pipeline nodes",
        )

    result = execute_pipeline(payload.algorithm_type, payload.sample_size)
    return result.model_dump()
