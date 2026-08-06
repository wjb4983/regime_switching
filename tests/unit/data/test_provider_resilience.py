"""Security and resilience contracts shared by data providers."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from regime.data.providers import (
    CachePolicy,
    CredentialScope,
    DataDomain,
    DateRange,
    MockProvider,
    Page,
    ProviderCredentials,
    ProviderRequest,
    RetryPolicy,
    SecurityIdentifier,
)
from regime.logging import REDACTION_TEXT, redact


def _request() -> ProviderRequest:
    return ProviderRequest(
        domain=DataDomain.EQUITY_OHLCV,
        identifiers=(SecurityIdentifier("SPY"),),
        date_range=DateRange(date(2024, 1, 1), date(2024, 1, 2)),
    )


def test_credentials_load_file_then_environment_and_enforce_scope(tmp_path, monkeypatch) -> None:
    secret_file = tmp_path / "secrets.local.json"
    secret_file.write_text(json.dumps({"API_KEY": "file-value"}), encoding="utf-8")
    monkeypatch.setenv("ACME_API_KEY", "environment-value")

    credentials = ProviderCredentials.load(
        env_prefix="ACME",
        secret_file=secret_file,
        required=("API_KEY",),
        scopes={"API_KEY": (CredentialScope.READ,)},
    )

    assert credentials.get("API_KEY", scope=CredentialScope.READ) == "environment-value"
    with pytest.raises(PermissionError, match="lacks 'write' scope"):
        credentials.get("API_KEY", scope=CredentialScope.WRITE)
    assert credentials.redacted()["API_KEY"] != "environment-value"


def test_loaded_secret_is_redacted_from_free_form_text(monkeypatch) -> None:
    monkeypatch.setenv("ACME_TOKEN", "unusually-shaped-private-value")
    ProviderCredentials.load(env_prefix="ACME", required=("TOKEN",))

    assert redact("failure: unusually-shaped-private-value") == f"failure: {REDACTION_TEXT}"


def test_request_cache_makes_fetch_idempotent() -> None:
    class CountingProvider(MockProvider):
        calls = 0

        def fetch(self, request: ProviderRequest) -> Page:  # type: ignore[type-arg]
            self.calls += 1
            return super().fetch(request)

    provider = CountingProvider()
    provider.fetch_page(_request())
    provider.fetch_page(_request())

    assert provider.calls == 1


def test_retry_exception_does_not_expose_credentials() -> None:
    secret = "vendor-key-with-a-distinct-value"
    credentials = ProviderCredentials(values={"API_KEY": secret})
    provider = MockProvider()
    provider.credentials = credentials
    provider.retry_policy = RetryPolicy(max_attempts=2, initial_backoff=timedelta(0))

    def fail() -> Page:
        raise OSError(f"upstream rejected {secret}")

    with pytest.raises(RuntimeError) as error:
        provider.request_with_retries(fail)
    assert secret not in str(error.value)
    assert REDACTION_TEXT in str(error.value)


def test_cache_can_be_disabled() -> None:
    class CountingProvider(MockProvider):
        calls = 0

        def fetch(self, request: ProviderRequest) -> Page:  # type: ignore[type-arg]
            self.calls += 1
            return super().fetch(request)

    provider = CountingProvider()
    provider.cache_policy = CachePolicy(enabled=False)
    provider.fetch_page(_request())
    provider.fetch_page(_request())
    assert provider.calls == 2
