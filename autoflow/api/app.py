from fastapi import FastAPI

from autoflow.api.routes_approvals import router as approvals_router


def create_app() -> FastAPI:
    app = FastAPI(title="AutoFlow", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(approvals_router)
    return app
