"""Server entry point.

FastAPI hosts both the dashboard and a small REST API. The dashboard satisfies
the evaluation workflow; the API exists because the brief asks for a reasonable
path to production integration, and a system that can only be driven by a human
in a browser does not have one.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import gradio as gr
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import get_settings
from app.pipeline import analyse_clip
from app.ratelimit import REGISTRY
from app.ui import build_ui

api = FastAPI(
    title="AutoAce Voice Tone & Noise Analysis",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


_basic = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials | None = Depends(_basic)) -> str:
    """HTTP basic auth for the REST API.

    The deployment is a public URL, so the API needs its own gate — the Gradio
    session cookie protects the dashboard but nothing else. Compared with
    `secrets.compare_digest` so a wrong username and a wrong password take the
    same time to reject.
    """
    import secrets

    settings = get_settings()
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok = secrets.compare_digest(credentials.username, settings.dashboard_user)
    pass_ok = secrets.compare_digest(credentials.password, settings.dashboard_password)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@api.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "privacy_mode": settings.privacy_mode.value,
        "tone_providers": settings.tone_providers,
        "providers_configured": {
            "gemini": bool(settings.gemini_api_key),
            "groq": bool(settings.groq_api_key),
        },
        "rate_limiters": REGISTRY.snapshot(),
    }


@api.post("/api/v1/analyze")
async def analyze(
    file: UploadFile = File(...),
    _user: str = Depends(require_auth),
) -> JSONResponse:
    """Analyse one clip and return the schema object.

    Kept deliberately simple: one file in, one result out. Batch work belongs in
    the dashboard, where progress and partial failure can be shown properly.
    """
    settings = get_settings()
    suffix = Path(file.filename or "clip").suffix or ".wav"

    workdir = Path(tempfile.mkdtemp(prefix="autoace_api_"))
    target = workdir / f"upload{suffix}"
    try:
        with target.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        size_mb = target.stat().st_size / (1 << 20)
        if size_mb > settings.max_file_mb:
            raise HTTPException(
                status_code=413,
                detail=f"file is {size_mb:.0f} MB, over the {settings.max_file_mb} MB limit",
            )

        report = await analyse_clip(target, settings)
        if report.status != "ok":
            return JSONResponse(
                status_code=422,
                content={"name": file.filename, "status": "failed", "error": report.error},
            )
        return JSONResponse({
            "name": file.filename,
            "status": "ok",
            "result": report.result.model_dump(mode="json"),
            "timings": report.timings,
        })
    finally:
        # The upload is deleted whether or not the analysis succeeded.
        shutil.rmtree(workdir, ignore_errors=True)


@api.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


settings = get_settings()
demo = build_ui()

# Gradio's built-in auth is the whole login story here: a shared credential
# handed to the evaluator, checked before the interface renders. It is the right
# weight for a single-tenant evaluation deployment; a multi-user product would
# need real identity.
app = gr.mount_gradio_app(
    api,
    demo,
    path="/dashboard",
    auth=(settings.dashboard_user, settings.dashboard_password),
    auth_message="Sign in with the credentials provided by AutoAce.",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        log_level="info",
    )
