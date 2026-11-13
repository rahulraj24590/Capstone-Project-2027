"""
Authentication & Role-Based Access Control (RBAC) Module (Phase 9).

Provides authentication and role permissions for LandslideGuard administration:
- Roles: 'admin' (full write/review privileges) and 'viewer' (read-only monitoring).
- Constant-time password hash comparison.
- Admin token validation for protected REST API routes.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default development administrator credentials
DEFAULT_ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
# SHA-256 of 'landslideguard2026'
DEFAULT_ADMIN_PASSWORD_HASH = os.getenv(
    "ADMIN_PASSWORD_HASH",
    hashlib.sha256("landslideguard2026".encode("utf-8")).hexdigest(),
)

# Admin API Secret Key for REST endpoints
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "landslideguard-admin-secret-2026")


def hash_password(password: str) -> str:
    """Hash a plaintext password using SHA-256."""
    return hashlib.sha256(password.strip().encode("utf-8")).hexdigest()


def authenticate_user(username: str, password: str) -> tuple[bool, str | None, str | None]:
    """Verify username and password against configured administrative credentials.

    Args:
        username: Provided username.
        password: Provided plaintext password.

    Returns:
        tuple (success: bool, role: str | None, error: str | None)
    """
    u = username.strip()
    p_hash = hash_password(password)

    if hmac.compare_digest(u, DEFAULT_ADMIN_USER) and hmac.compare_digest(p_hash, DEFAULT_ADMIN_PASSWORD_HASH):
        return True, "admin", None

    # Optional viewer guest login
    if u.lower() in ("viewer", "guest") and (password.strip() == "" or password.strip().lower() == "guest"):
        return True, "viewer", None

    return False, None, "Invalid username or password. Please verify credentials."


def verify_admin_key(key_or_token: str | None) -> bool:
    """Verify whether a provided API key or Bearer token matches the administrative secret."""
    if not key_or_token:
        return False
    clean = key_or_token.strip()
    if clean.startswith("Bearer "):
        clean = clean[7:].strip()
    return hmac.compare_digest(clean, ADMIN_API_KEY)


def can_manage_locations(role: str | None) -> bool:
    """Check whether role has permission to create or pause monitoring stations."""
    return role == "admin"


def can_review_community_reports(role: str | None) -> bool:
    """Check whether role has permission to verify or reject community reports."""
    return role == "admin"


def can_manage_alerts(role: str | None) -> bool:
    """Check whether role has permission to acknowledge or resolve alerts."""
    return role == "admin"

