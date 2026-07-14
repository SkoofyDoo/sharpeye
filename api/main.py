"""SharpEye REST API - FastAPI application"""

from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routes.v1 import router as v1_router

limiter = Limiter(key_func = get_remote_address)

app = FastAPI(
    title = "SharpEye API",
    description = "Image quality control - human-readable verdicts for people and agents",
    version = "0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(v1_router)


@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request) -> dict:
    return {"status": "ok", "service": "sharpeye"}
