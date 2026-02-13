"""
API v1 router aggregating all route modules.
"""

from fastapi import APIRouter

from app.api.v1.routes_auth import router as auth_router
from app.api.v1.routes_chat import router as chat_router
from app.api.v1.routes_chat_streaming import router as chat_streaming_router
from app.api.v1.routes_health import router as health_router
from app.api.v1.routes_rag import router as rag_router

# Create main v1 router
api_router = APIRouter()

# Include all route modules
api_router.include_router(health_router, tags=["health"])
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(chat_streaming_router, tags=["chat"])  # Streaming endpoint
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])

# Additional routers will be added in subsequent phases:
# api_router.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
# api_router.include_router(memory_router, prefix="/memory", tags=["memory"])
# api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
# api_router.include_router(vertex_router, prefix="/vertex", tags=["vertex"])
# api_router.include_router(image_router, prefix="/image", tags=["image"])
# api_router.include_router(usage_router, prefix="/usage", tags=["usage"])
# api_router.include_router(github_router, prefix="/github", tags=["github"])
# api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
# api_router.include_router(payments_router, prefix="/payments", tags=["payments"])
