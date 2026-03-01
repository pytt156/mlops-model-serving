from fastapi import FastAPI

app = FastAPI(title="CIFAR-10 Prediction API")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check for Docker/orchestration."""
    return {"status": "ok"}
