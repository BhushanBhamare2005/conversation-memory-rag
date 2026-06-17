from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Intelligent Conversational Memory System", version="1.0.0")
app.include_router(router)
