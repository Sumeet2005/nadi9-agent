from fastapi import FastAPI

from nadi9.api.routes.health import router as health_router
from nadi9.api.routes.reviews import router as reviews_router
from nadi9.api.routes.runs import router as runs_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI production application."""
    app = FastAPI(
        title="NADI-9 Subtitle Decision API",
        description="Evidence-Grounded Agentic Subtitle Decision API",
        version="0.1.0",
    )

    app.include_router(health_router)
    app.include_router(runs_router)
    app.include_router(reviews_router)

    return app


app = create_app()
