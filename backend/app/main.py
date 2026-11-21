"""
Main FastAPI Application Entry Point.
Serves both the REST API (/api/v1/*) and the compiled React SPA (frontend/dist) in production.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.api.endpoints import router as api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Research REST API for Parkinson's Disease classification from spiral drawings "
        "using Deep Learning architectures (Custom CNN, VGG16, ResNet50)."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local React/Vite development and multi-platform testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve EDA and Evaluation plots statically
for path_dir in [settings.EDA_DIR, settings.EVAL_DIR, settings.EXPLAIN_DIR]:
    os.makedirs(path_dir, exist_ok=True)

app.mount("/static/eda", StaticFiles(directory=settings.EDA_DIR), name="eda_plots")
app.mount("/static/evaluation", StaticFiles(directory=settings.EVAL_DIR), name="evaluation_plots")

# Serve compiled React Frontend in production if frontend/dist exists
FRONTEND_DIST = os.path.join(settings.BASE_DIR, "frontend", "dist")
FRONTEND_ASSETS = os.path.join(FRONTEND_DIST, "assets")
FRONTEND_INDEX = os.path.join(FRONTEND_DIST, "index.html")

if os.path.isdir(FRONTEND_ASSETS):
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="frontend_assets")


@app.get("/")
async def root():
    if os.path.isfile(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health",
        "models_endpoint": f"{settings.API_V1_STR}/models",
        "metrics_endpoint": f"{settings.API_V1_STR}/metrics",
        "medical_disclaimer": settings.MEDICAL_DISCLAIMER
    }


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if full_path.startswith(("api/", "static/", "docs", "redoc", "openapi.json")):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    candidate = os.path.join(FRONTEND_DIST, full_path)
    if os.path.isfile(candidate):
        return FileResponse(candidate)
    if os.path.isfile(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=True)
