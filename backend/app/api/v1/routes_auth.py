"""
Authentication API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.exceptions import (
    AuthenticationError,
    InvalidInviteCodeError,
    UserNotFoundError,
)
from app.db.models.user import User
from app.services.auth_service import AuthService

router = APIRouter()


class RegisterRequest(BaseModel):
    """Register request schema."""

    telegram_id: int = Field(..., description="Telegram user ID")
    username: str | None = Field(None, description="Telegram username")
    first_name: str | None = Field(None, description="First name")
    last_name: str | None = Field(None, description="Last name")


class RegisterResponse(BaseModel):
    """Register response schema."""

    user_id: int
    telegram_id: int
    role: str
    authorized: bool
    token: str


class UnlockRequest(BaseModel):
    """Unlock request schema."""

    telegram_id: int
    unlock_code: str


class UnlockResponse(BaseModel):
    """Unlock response schema."""

    user_id: int
    telegram_id: int
    role: str
    authorized: bool
    message: str


class BootstrapRequest(BaseModel):
    """Bootstrap admin request schema."""

    telegram_id: int
    bootstrap_code: str


class BootstrapResponse(BaseModel):
    """Bootstrap response schema."""

    user_id: int
    telegram_id: int
    role: str
    message: str


class InviteRequest(BaseModel):
    """Invite code consumption request."""

    telegram_id: int
    invite_code: str


class InviteResponse(BaseModel):
    """Invite response schema."""

    user_id: int
    telegram_id: int
    role: str
    message: str


class UserResponse(BaseModel):
    """User response schema."""

    user_id: int
    telegram_id: int
    username: str | None
    first_name: str | None
    last_name: str | None
    role: str
    authorized: bool
    verified: bool
    subscription_tier: str | None
    default_mode: str
    settings: dict


@router.post("/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> RegisterResponse:
    """
    Register a new user or get existing user.

    Creates user with DEMO role and returns JWT token.
    """
    auth_service = AuthService(db)

    try:
        user, token = await auth_service.register(
            telegram_id=request.telegram_id,
            username=request.username,
            first_name=request.first_name,
            last_name=request.last_name,
        )

        return RegisterResponse(
            user_id=user.id,
            telegram_id=user.telegram_id,
            role=user.role,
            authorized=user.authorized,
            token=token,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd rejestracji: {str(e)}",
        ) from e


@router.post("/unlock", response_model=UnlockResponse)
async def unlock(
    request: UnlockRequest,
    db: AsyncSession = Depends(get_db),
) -> UnlockResponse:
    """
    Unlock DEMO access for a user.

    Requires valid DEMO unlock code.
    """
    auth_service = AuthService(db)

    try:
        user = await auth_service.unlock(
            telegram_id=request.telegram_id,
            unlock_code=request.unlock_code,
        )

        return UnlockResponse(
            user_id=user.id,
            telegram_id=user.telegram_id,
            role=user.role,
            authorized=user.authorized,
            message="Dostęp DEMO odblokowany pomyślnie",
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Użytkownik nie istnieje. Najpierw zarejestruj się.",
        ) from e
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e


@router.post("/bootstrap", response_model=BootstrapResponse)
async def bootstrap(
    request: BootstrapRequest,
    db: AsyncSession = Depends(get_db),
) -> BootstrapResponse:
    """
    Bootstrap admin user.

    Requires valid admin bootstrap code.
    """
    auth_service = AuthService(db)

    try:
        user = await auth_service.bootstrap(
            telegram_id=request.telegram_id,
            bootstrap_code=request.bootstrap_code,
        )

        return BootstrapResponse(
            user_id=user.id,
            telegram_id=user.telegram_id,
            role=user.role,
            message="Administrator utworzony pomyślnie",
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Użytkownik nie istnieje. Najpierw zarejestruj się.",
        ) from e
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e


@router.post("/invite", response_model=InviteResponse)
async def consume_invite(
    request: InviteRequest,
    db: AsyncSession = Depends(get_db),
) -> InviteResponse:
    """
    Consume invite code to upgrade role.
    """
    auth_service = AuthService(db)

    try:
        user = await auth_service.consume_invite(
            telegram_id=request.telegram_id,
            invite_code=request.invite_code,
        )

        return InviteResponse(
            user_id=user.id,
            telegram_id=user.telegram_id,
            role=user.role,
            message=f"Kod zaproszenia wykorzystany. Nowa rola: {user.role}",
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Użytkownik nie istnieje",
        ) from e
    except InvalidInviteCodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """
    Get current authenticated user.

    Requires valid JWT token in Authorization header.
    """
    return UserResponse(
        user_id=current_user.id,
        telegram_id=current_user.telegram_id,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        role=current_user.role,
        authorized=current_user.authorized,
        verified=current_user.verified,
        subscription_tier=current_user.subscription_tier,
        default_mode=current_user.default_mode,
        settings=current_user.settings,
    )


class UpdateSettingsRequest(BaseModel):
    """Update settings request schema."""

    default_mode: str | None = None
    settings: dict | None = None


@router.put("/settings", response_model=UserResponse)
async def update_settings(
    request: UpdateSettingsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Update user settings.
    """
    if request.default_mode:
        current_user.default_mode = request.default_mode

    if request.settings:
        current_user.settings = request.settings

    await db.commit()
    await db.refresh(current_user)

    return UserResponse(
        user_id=current_user.id,
        telegram_id=current_user.telegram_id,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        role=current_user.role,
        authorized=current_user.authorized,
        verified=current_user.verified,
        subscription_tier=current_user.subscription_tier,
        default_mode=current_user.default_mode,
        settings=current_user.settings,
    )
