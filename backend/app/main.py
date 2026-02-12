from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .db import engine
from .models import Base
from .routers.auth import router as auth_router
from .routers.admin import router as admin_router
from .routers.dataprocessor_profiles import router as dataproc_profiles_router
from .routers.profiles import router as profiles_router, seed_public_profiles
from .routers.runs import router as runs_router
from .routers.videos import router as videos_router
from .services.storage import ensure_dirs


settings = Settings()

app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_dirs()
    Base.metadata.create_all(bind=engine)
    from .db import SessionLocal

    db = SessionLocal()
    try:
        seed_public_profiles(db)
        db.commit()
    finally:
        db.close()


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(videos_router)
app.include_router(profiles_router)
app.include_router(dataproc_profiles_router)
app.include_router(runs_router)

