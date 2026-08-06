"""Massive REST adapter for historical stock and option aggregates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlparse
from urllib.request import Request, urlopen

from regime.data.providers.base import ProviderCredentials


class MassiveAPIError(RuntimeError):
    """Raised when the Massive API returns a non-success payload."""


@dataclass(frozen=True, slots=True)
class MassivePlan:
    """Plan limits used to derive the maximum downloadable history window."""

    stocks_history_years: int | None
    options_history_years: int | None


PLANS: dict[str, MassivePlan] = {
    "basic": MassivePlan(stocks_history_years=2, options_history_years=2),
    "starter": MassivePlan(stocks_history_years=5, options_history_years=2),
    "developer": MassivePlan(stocks_history_years=10, options_history_years=4),
    "advanced": MassivePlan(stocks_history_years=None, options_history_years=None),
}

STOCKS_ALL_HISTORY_START = date(2003, 9, 10)
OPTIONS_ALL_HISTORY_START = date(2014, 6, 2)


class MassiveClient:
    """Small JSON client for Massive REST endpoints."""

    def __init__(
        self,
        *,
        credentials: ProviderCredentials,
        base_url: str = "https://api.massive.com",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_api_key(
        cls,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.massive.com",
        secret_file: str | None = None,
    ) -> "MassiveClient":
        credentials = ProviderCredentials.load(
            env_prefix="MASSIVE",
            secret_file=secret_file,
            required=() if api_key else ("API_KEY",),
            names=(),
        )
        if api_key:
            credentials = ProviderCredentials(values={"API_KEY": api_key}, secret_keys=("API_KEY",))
        return cls(credentials=credentials, base_url=base_url)

    def fetch_stock_aggregates(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        multiplier: int = 1,
        timespan: str = "day",
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50_000,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            (
                f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
                f"{start_date.isoformat()}/{end_date.isoformat()}"
            ),
            {"adjusted": str(adjusted).lower(), "sort": sort, "limit": str(limit)},
        )
        return self._collect_results(payload)

    def fetch_option_contracts(
        self,
        *,
        underlying_symbol: str,
        as_of_date: date,
        expired: bool = True,
        limit: int = 1_000,
        contract_type: str | None = None,
        expiration_date_gte: date | None = None,
        expiration_date_lte: date | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {
            "underlying_ticker": underlying_symbol,
            "as_of": as_of_date.isoformat(),
            "expired": str(expired).lower(),
            "order": "asc",
            "sort": "ticker",
            "limit": str(limit),
        }
        if contract_type:
            params["contract_type"] = contract_type
        if expiration_date_gte is not None:
            params["expiration_date.gte"] = expiration_date_gte.isoformat()
        if expiration_date_lte is not None:
            params["expiration_date.lte"] = expiration_date_lte.isoformat()
        payload = self._get_json("/v3/reference/options/contracts", params)
        return self._collect_results(payload, allow_link_pagination=True)

    def fetch_option_aggregates(
        self,
        *,
        option_ticker: str,
        start_date: date,
        end_date: date,
        multiplier: int = 1,
        timespan: str = "day",
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50_000,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            (
                f"/v2/aggs/ticker/{option_ticker}/range/{multiplier}/{timespan}/"
                f"{start_date.isoformat()}/{end_date.isoformat()}"
            ),
            {"adjusted": str(adjusted).lower(), "sort": sort, "limit": str(limit)},
        )
        return self._collect_results(payload)

    def _collect_results(
        self, payload: Mapping[str, Any], *, allow_link_pagination: bool = False
    ) -> list[dict[str, Any]]:
        records = [dict(item) for item in payload.get("results", [])]
        next_url = payload.get("next_url")
        while isinstance(next_url, str) and next_url:
            page = self._get_json_from_url(next_url)
            records.extend(dict(item) for item in page.get("results", []))
            next_url = page.get("next_url")
        if allow_link_pagination and isinstance(payload.get("_link_next_url"), str):
            next_url = payload["_link_next_url"]
            while next_url:
                page = self._get_json_from_url(next_url)
                records.extend(dict(item) for item in page.get("results", []))
                next_url = page.get("_link_next_url")
        return records

    def _get_json(self, path: str, params: Mapping[str, str]) -> dict[str, Any]:
        query = dict(params)
        query["apiKey"] = self._api_key()
        url = f"{self._base_url}{path}?{urlencode(query)}"
        return self._request_json(url)

    def _get_json_from_url(self, url: str) -> dict[str, Any]:
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["apiKey"] = self._api_key()
        safe_url = parsed._replace(query=urlencode(query)).geturl()
        return self._request_json(safe_url)

    def _request_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
                next_url = _extract_next_link(dict(response.headers))
                if next_url:
                    payload["_link_next_url"] = next_url
        except HTTPError as exc:
            raise MassiveAPIError(
                f"MASSIVE HTTP {exc.code} for {_sanitize_url(exc.url or url)}"
            ) from None
        if payload.get("status") not in {None, "OK"}:
            raise MassiveAPIError(str(payload))
        return payload

    def _api_key(self) -> str:
        return self._credentials.get("API_KEY").strip()


def _extract_next_link(headers: Mapping[str, str]) -> str | None:
    link_header = headers.get("Link") or headers.get("link")
    if not link_header:
        return None
    for part in link_header.split(","):
        candidate = part.strip()
        if 'rel="next"' not in candidate and "rel=next" not in candidate:
            continue
        start = candidate.find("<")
        end = candidate.find(">", start + 1)
        if start != -1 and end != -1:
            return candidate[start + 1 : end]
    return None


def _sanitize_url(url: str) -> str:
    parsed = urlparse(url)
    query = [
        (key, "[REDACTED]" if key == "apiKey" else value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return parsed._replace(query=urlencode(query)).geturl()
