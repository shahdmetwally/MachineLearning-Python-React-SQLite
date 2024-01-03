from .app import app
from .admin import router
from .user import router

app.include_router(router, prefix="/api")

# uvicorn Model.server.main:app --host 0.0.0.0 --port 8000
