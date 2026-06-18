from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Intelligent Conversational Memory System", version="1.0.0")
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
	return {
		"status": "ok",
		"service": "Intelligent Conversational Memory System",
		"health": "/health",
		"chat": "/chat",
		"docs": "/docs",
	}
