"""`python -m agentplatform` -> run the API server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("agentplatform.api.app:create_app", factory=True, host="0.0.0.0", port=8000)
