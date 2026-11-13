"""
Optional external notification provider interfaces for LandslideGuard.

Safety guarantee:
- Email and SMS notifications are STRICTLY DISABLED by default.
- No external network/email/SMS dispatch ever occurs unless explicitly enabled
  in configuration AND required environment credentials are supplied.
"""

from __future__ import annotations

import abc
import os
from typing import Any

from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)


class NotificationProvider(abc.ABC):
    """Abstract base class for future alert notification channels."""

    channel: str = "base"

    @abc.abstractmethod
    def is_enabled(self) -> bool:
        """Return True only if explicitly enabled in configuration."""
        ...

    @abc.abstractmethod
    def is_configured(self) -> bool:
        """Return True only when required credentials/targets exist in env."""
        ...

    @abc.abstractmethod
    def send(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Attempt dispatch or report disabled/unconfigured state cleanly."""
        ...


class EmailNotificationProvider(NotificationProvider):
    """Prepared SMTP/Email notification interface (disabled by default)."""

    channel = "email"

    def is_enabled(self) -> bool:
        cfg = get_setting("alerts", "notifications", "email") or {}
        return bool(cfg.get("enabled", False))

    def is_configured(self) -> bool:
        return bool(
            os.environ.get("ALERT_SMTP_HOST")
            and os.environ.get("ALERT_EMAIL_TO")
        )

    def send(self, alert: dict[str, Any]) -> dict[str, Any]:
        if not self.is_enabled():
            return {
                "channel": self.channel,
                "dispatched": False,
                "reason": "disabled_by_default",
            }
        if not self.is_configured():
            return {
                "channel": self.channel,
                "dispatched": False,
                "reason": "missing_env_credentials (ALERT_SMTP_HOST / ALERT_EMAIL_TO)",
            }
        # Placeholder integration hook; no live SMTP transport invoked unless implemented
        logger.info(
            "Email notification hook ready for alert #%s (%s) — external send requires SMTP adapter.",
            alert.get("id"),
            alert.get("risk_category"),
        )
        return {
            "channel": self.channel,
            "dispatched": False,
            "reason": "hook_prepared_only",
        }


class SMSNotificationProvider(NotificationProvider):
    """Prepared SMS gateway interface (disabled by default)."""

    channel = "sms"

    def is_enabled(self) -> bool:
        cfg = get_setting("alerts", "notifications", "sms") or {}
        return bool(cfg.get("enabled", False))

    def is_configured(self) -> bool:
        return bool(
            os.environ.get("ALERT_SMS_API_KEY")
            and os.environ.get("ALERT_SMS_TO")
        )

    def send(self, alert: dict[str, Any]) -> dict[str, Any]:
        if not self.is_enabled():
            return {
                "channel": self.channel,
                "dispatched": False,
                "reason": "disabled_by_default",
            }
        if not self.is_configured():
            return {
                "channel": self.channel,
                "dispatched": False,
                "reason": "missing_env_credentials (ALERT_SMS_API_KEY / ALERT_SMS_TO)",
            }
        logger.info(
            "SMS notification hook ready for alert #%s (%s) — external send requires gateway adapter.",
            alert.get("id"),
            alert.get("risk_category"),
        )
        return {
            "channel": self.channel,
            "dispatched": False,
            "reason": "hook_prepared_only",
        }


class NotificationDispatcher:
    """Routes experimental alert payloads to enabled notification providers."""

    def __init__(self, providers: list[NotificationProvider] | None = None) -> None:
        self.providers: list[NotificationProvider] = providers or [
            EmailNotificationProvider(),
            SMSNotificationProvider(),
        ]

    def get_channel_status(self) -> list[dict[str, Any]]:
        """Report readiness of all notification channels without sending anything."""
        return [
            {
                "channel": p.channel,
                "enabled": p.is_enabled(),
                "configured": p.is_configured(),
            }
            for p in self.providers
        ]

    def dispatch(self, alert: dict[str, Any]) -> list[dict[str, Any]]:
        """Invoke each configured provider, skipping disabled channels cleanly."""
        results = []
        for provider in self.providers:
            try:
                results.append(provider.send(alert))
            except Exception as exc:
                logger.warning("Notification channel '%s' failed cleanly: %s", provider.channel, exc)
                results.append({
                    "channel": provider.channel,
                    "dispatched": False,
                    "reason": f"error: {exc}",
                })
        return results

