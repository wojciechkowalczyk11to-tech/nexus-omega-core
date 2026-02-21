"""
Custom exception hierarchy for the application.
All exceptions include Polish error messages for user-facing errors.
"""

from typing import Any


class AppException(Exception):  # noqa: N818
    """Base exception for all application errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class PolicyDeniedError(AppException):
    """Raised when user access is denied by policy engine."""

    def __init__(self, reason: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Dostęp zabroniony: {reason}", details)
        self.reason = reason


class ProviderError(AppException):
    """Raised when a provider fails to generate a response."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Błąd providera: {message}", details)
        self.provider = details.get("provider", "unknown") if details else "unknown"


class AllProvidersFailedError(AppException):
    """Raised when all providers in the fallback chain fail."""

    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        super().__init__(
            "Wszystkie providery AI zawiodły. Spróbuj ponownie później.",
            {"attempts": attempts},
        )
        self.attempts = attempts


class BudgetExceededError(AppException):
    """Raised when user exceeds their daily budget."""

    def __init__(self, current: float, limit: float) -> None:
        super().__init__(
            f"Przekroczono dzienny limit budżetu: ${current:.2f} / ${limit:.2f}",
            {"current": current, "limit": limit},
        )
        self.current = current
        self.limit = limit


class RateLimitError(AppException):
    """Raised when user exceeds rate limit."""

    def __init__(self, limit: int, window: str = "minute") -> None:
        super().__init__(
            f"Przekroczono limit zapytań: {limit} na {window}. Spróbuj ponownie za chwilę.",
            {"limit": limit, "window": window},
        )
        self.limit = limit
        self.window = window


class AuthenticationError(AppException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Błąd uwierzytelniania") -> None:
        super().__init__(message)


class AuthorizationError(AppException):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Brak uprawnień") -> None:
        super().__init__(message)


class UserNotFoundError(AppException):
    """Raised when user is not found in database."""

    def __init__(self, telegram_id: int) -> None:
        super().__init__(
            f"Użytkownik {telegram_id} nie został znaleziony",
            {"telegram_id": telegram_id},
        )
        self.telegram_id = telegram_id


class SessionNotFoundError(AppException):
    """Raised when chat session is not found."""

    def __init__(self, session_id: int) -> None:
        super().__init__(
            f"Sesja {session_id} nie została znaleziona",
            {"session_id": session_id},
        )
        self.session_id = session_id


class InvalidInviteCodeError(AppException):
    """Raised when invite code is invalid or expired."""

    def __init__(self, message: str = "Kod zaproszenia jest nieprawidłowy lub wygasł") -> None:
        super().__init__(message)


class PaymentError(AppException):
    """Raised when payment processing fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Błąd płatności: {message}", details)


class SubscriptionExpiredError(AppException):
    """Raised when user's subscription has expired."""

    def __init__(self, message: str = "Twoja subskrypcja wygasła") -> None:
        super().__init__(message)


class RAGError(AppException):
    """Raised when RAG operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Błąd RAG: {message}", details)


class VertexSearchError(AppException):
    """Raised when Vertex AI Search fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Błąd wyszukiwania Vertex: {message}", details)


class GitHubError(AppException):
    """Raised when GitHub operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Błąd GitHub: {message}", details)


class ToolExecutionError(AppException):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Błąd narzędzia {tool_name}: {message}", details)
        self.tool_name = tool_name


class ValidationError(AppException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None) -> None:
        details = {"field": field} if field else {}
        super().__init__(f"Błąd walidacji: {message}", details)
        self.field = field


class DatabaseError(AppException):
    """Raised when database operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Błąd bazy danych: {message}", details)


class CacheError(AppException):
    """Raised when cache operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Błąd cache: {message}", details)


class SandboxError(AppException):
    """Raised when sandbox execution fails."""

    def __init__(
        self, message: str = "Błąd wykonania sandbox", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, details)
