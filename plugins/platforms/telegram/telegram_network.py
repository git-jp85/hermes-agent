"""Telegram network helpers: a hostname-preserving fallback transport (Host + SNI stay
api.telegram.org while TCP retries known IPv4 literals) plus DoH-based IP discovery."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import ssl
from typing import Iterable, Optional

import httpx

logger = logging.getLogger(__name__)

_TELEGRAM_API_HOST = "api.telegram.org"


def _describe_transport_error(error: Exception) -> str:
    """Return a non-empty, secret-safe exception representation for diagnostics."""
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(repr(error), force=True)
    except Exception:
        return type(error).__name__


def _classify_transport_error(error: Exception) -> str:
    """Bucket a transport exception into a small, log-friendly kind label (seq13 observability)."""
    if isinstance(
        error,
        (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return "timeout"
    if isinstance(error, (httpx.ConnectError, ConnectionError)):
        return "connect"
    if isinstance(error, httpx.ReadError):
        return "read"
    text = str(error).lower()
    if "wrong_version" in text or "ssl" in text:
        return "tls"
    return type(error).__name__

# TCP keepalive so a half-open/CLOSE-WAIT long-poll errors out instead of blocking getUpdates forever
# (Windows leaves SO_KEEPALIVE off). Idle/interval knobs are best-effort per Python/OS combo.
# Windows does not enable SO_KEEPALIVE on new sockets by default, so a dead api.telegram.org peer can hang
# forever (#87057).
_TCP_KEEPALIVE_IDLE_S = 30
_TCP_KEEPALIVE_INTERVAL_S = 10
_TCP_KEEPALIVE_COUNT = 3


def tcp_keepalive_socket_options() -> list[tuple[int, int, int]]:
    """``setsockopt`` tuples for httpx ``socket_options``: always SO_KEEPALIVE, plus idle/interval/count
    when the interpreter exposes those option names."""
    options: list[tuple[int, int, int]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)
    for opt, value in ((idle, _TCP_KEEPALIVE_IDLE_S), (getattr(socket, "TCP_KEEPINTVL", None), _TCP_KEEPALIVE_INTERVAL_S),
                       (getattr(socket, "TCP_KEEPCNT", None), _TCP_KEEPALIVE_COUNT)):
        if opt is not None:
            options.append((socket.IPPROTO_TCP, opt, value))
    return options

# DNS-over-HTTPS providers: discover Telegram API IPs the (possibly unreachable) local resolver may not
# return. Bounded so connect() isn't delayed.
_DOH_TIMEOUT = 4.0
_DOH_PROVIDERS: list[dict] = [
    {"url": "https://dns.google/resolve", "params": {"name": _TELEGRAM_API_HOST, "type": "A"}, "headers": {}},
    {
        "url": "https://cloudflare-dns.com/dns-query", "params": {"name": _TELEGRAM_API_HOST, "type": "A"},
        "headers": {"Accept": "application/dns-json"}},
]
# Last-resort IPv4 Bot API endpoints (149.154.160.0/20). Used when DoH is blocked AND as
# first-try connect targets so a blackholed IPv6 AAAA for the hostname can't pin initialize().
SEED_FALLBACK_IPS: list[str] = ["149.154.166.110", "149.154.167.220"]
# MTProto-only Telegram data-centre addresses: same Telegram-owned bands as the Bot API, they accept
# TCP :443 but speak MTProto instead of Bot-API TLS, so every request routed to them dies with
# SSL WRONG_VERSION_NUMBER. They leak into the candidate list through DoH A-records (Google/Cloudflare
# answer api.telegram.org with DC addresses too): during the 2026-10-05 incident 149.154.175.50
# (76 attempts) and 91.108.56.130 (70 attempts) accounted for every WRONG_VERSION error. Dropping them
# before anything dials them only saves the wasted handshake - the TLS probe stays the authoritative
# filter, because no hand-maintained list can be complete.
_MT_PROTO_ONLY_IPS: frozenset[str] = frozenset({"149.154.175.50", "91.108.56.130"})
# Demoted to the back of the candidate order. 149.154.167.220 is the system-DNS address the sticky path
# pinned during the 2026-10-05/06 incident (3,777 attempts, sticky failures) while 149.154.166.110
# served 8,942. It stays in the list as a last resort but never ahead of an equivalent literal, even
# when it is the sticky pick.
_LAST_RESORT_IPS: frozenset[str] = frozenset({"149.154.167.220"})
# The Bot API is only served from Telegram-owned networks. Both discovery legs (system resolver and
# DoH) must stay inside them: a hijacked/poisoned DNS answer or an unrelated A record that happens to
# be returned for api.telegram.org must never become a connect target (#87015).
_TELEGRAM_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("149.154.160.0/20"),
    ipaddress.ip_network("91.108.0.0/16"),
)
_UNSET = object()


def _resolve_proxy_url(target_hosts=None) -> str | None:
    from gateway.platforms.base import resolve_proxy_url  # env vars + macOS system proxy
    return resolve_proxy_url("TELEGRAM_PROXY", target_hosts=target_hosts)


class TelegramFallbackTransport(httpx.AsyncBaseTransport):
    """Reach the Bot API via known IPv4 literals first, dual-stack hostname last. Host + SNI stay on
    api.telegram.org (like ``curl --resolve``) so a blackholed IPv6 AAAA can't pin initialize()."""

    # Bound every pool: httpx's 100-connection default × (wedged endpoint + seed IPs) can outgrow the fd limit.
    # See #63311.
    _POOL_LIMITS = httpx.Limits(max_connections=8, max_keepalive_connections=4)

    def __init__(self, fallback_ips: Iterable[str], **transport_kwargs):
        self._fallback_ips = list(dict.fromkeys(_normalize_fallback_ips(fallback_ips)))
        proxy_url = _resolve_proxy_url(target_hosts=[_TELEGRAM_API_HOST, *self._fallback_ips])
        if proxy_url and "proxy" not in transport_kwargs:
            transport_kwargs["proxy"] = proxy_url
        transport_kwargs.setdefault("limits", self._POOL_LIMITS)
        transport_kwargs.setdefault("socket_options", tcp_keepalive_socket_options())
        self._transport_kwargs = transport_kwargs
        self._primary = httpx.AsyncHTTPTransport(**transport_kwargs)
        self._primary_lock = asyncio.Lock()
        self._primary_closed = False
        # Built on demand and discarded on failure — see _reset_fallback.
        self._fallbacks: dict[str, httpx.AsyncHTTPTransport] = {}
        self._fallback_lock = asyncio.Lock()
        # ``_UNSET`` / ``None`` / ``str`` = no sticky yet / sticky hostname / sticky IPv4.
        self._sticky_ip: object = _UNSET
        self._sticky_lock = asyncio.Lock()
        self._last_failure: tuple[str, str] | None = None
        # Consecutive connect failures per path (key: IPv4 str or None=hostname). A path that failed
        # on the previous walk is deprioritised so the transport stops re-hitting a blackholed
        # endpoint ahead of a freshly healthy one (failover acceleration, seq13).
        self._path_failures: dict[Optional[str], int] = {}
        # Failure-kind counters for the one-line transport-health summary logged on recovery.
        self._failure_kinds: dict[str, int] = {}

    async def _get_fallback(self, ip: str) -> httpx.AsyncHTTPTransport:
        async with self._fallback_lock:
            transport = self._fallbacks.get(ip)
            if transport is None:
                transport = httpx.AsyncHTTPTransport(**self._transport_kwargs)
                self._fallbacks[ip] = transport
            return transport

    async def _reset_primary(self, transport: httpx.AsyncHTTPTransport) -> None:
        # Retryable primary failures leave half-closed sockets in the pool; replace the generation first.
        async with self._primary_lock:
            if self._primary_closed or transport is not self._primary:
                return
            self._primary = httpx.AsyncHTTPTransport(**self._transport_kwargs)
        try:
            await transport.aclose()
        except Exception as exc:
            logger.debug("[Telegram] Error closing primary transport: %s", exc)

    async def _reset_fallback(self, ip: str) -> None:
        """Discard a failed fallback pool: a peer-closed connect leaves a CLOSE_WAIT socket in it, and the
        poisoned pool would leak one fd per retry.

        Retaining the poisoned pool leaks one descriptor per retry until the process hits its file limit and
        can no longer accept connections or resolve DNS (#63311).
        """
        async with self._fallback_lock:
            transport = self._fallbacks.pop(ip, None)
        if transport is None:
            return
        try:
            await transport.aclose()
        except Exception as exc:  # closing a broken pool must never mask the real error
            logger.debug("[Telegram] Error closing fallback transport %s: %s", ip, exc)

    def _attempt_order(self) -> list[Optional[str]]:
        """Sticky path first, then IPv4 literals, dual-stack hostname last (a blackholed IPv6 path never
        errors — Happy Eyeballs waits on AAAA until the OS TCP timeout and can pin the loop)."""
        order: list[Optional[str]] = []
        if self._sticky_ip is not _UNSET:
            order.append(None if self._sticky_ip is None else str(self._sticky_ip))
        order.extend(ip for ip in self._fallback_ips if ip not in order)
        # Deprioritise paths that failed on the previous walk (stable sort keeps config order for
        # ties) so a blackholed endpoint is tried after a freshly healthy one.
        tail = [ip for ip in order if ip is not None]
        # Two keys, most significant first: last-resort literals always behind everything else (even
        # when sticky selected one), then paths that failed on the previous walk (stable sort keeps
        # config order for ties) so a blackholed endpoint is tried after a freshly healthy one.
        tail.sort(key=lambda ip: (ip in _LAST_RESORT_IPS, self._path_failures.get(ip, 0)))
        order = [ip for ip in order if ip is None] + tail
        if None not in order:
            order.append(None)
        return order

    def _failure_summary(self) -> str:
        """Compact 'kind=count' tally of transport failures since start, for one-line health logs."""
        if not self._failure_kinds:
            return "none"
        return ", ".join(
            f"{kind}={count}"
            for kind, count in sorted(self._failure_kinds.items(), key=lambda kv: -kv[1])
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != _TELEGRAM_API_HOST or not self._fallback_ips:
            return await self._primary.handle_async_request(request)
        last_error: Exception | None = None
        for ip in self._attempt_order():
            candidate = request if ip is None else _rewrite_request_for_ip(request, ip)
            transport = self._primary if ip is None else await self._get_fallback(ip)
            try:
                response = await transport.handle_async_request(candidate)
                # Success clears this path's failure streak.
                self._path_failures.pop(ip, None)
                if self._last_failure is not None:
                    failed_path, failure = self._last_failure
                    self._last_failure = None
                    logger.info(
                        "[Telegram] Telegram API transport recovered via %s after %s failed: %s "
                        "(transport failure tally: %s)",
                        ip or _TELEGRAM_API_HOST,
                        failed_path,
                        failure,
                        self._failure_summary(),
                    )
                if self._sticky_ip is _UNSET or self._sticky_ip != ip:
                    async with self._sticky_lock:
                        if self._sticky_ip is _UNSET or self._sticky_ip != ip:
                            self._sticky_ip = ip
                            if ip is not None:
                                log = logger.warning if last_error is not None else logger.info
                                log("[Telegram] Using sticky IPv4 Telegram API path %s (dual-stack hostname tried last — #87015)", ip)
                return response
            except Exception as exc:
                last_error = exc
                if not _is_retryable_connect_error(exc):
                    raise
                path = ip or _TELEGRAM_API_HOST
                failure = _describe_transport_error(exc)
                self._failure_kinds[_classify_transport_error(exc)] = (
                    self._failure_kinds.get(_classify_transport_error(exc), 0) + 1
                )
                self._path_failures[ip] = self._path_failures.get(ip, 0) + 1
                self._last_failure = (path, failure)
                if self._sticky_ip is not _UNSET and ip == self._sticky_ip:
                    async with self._sticky_lock:
                        if self._sticky_ip is not _UNSET and self._sticky_ip == ip:
                            self._sticky_ip = _UNSET
                            logger.warning(
                                "[Telegram] Sticky Telegram path %s failed; re-walking IPv4 literals before the hostname",
                                ip if ip is not None else "api.telegram.org")
                if ip is None:
                    await self._reset_primary(transport)
                    logger.warning("[Telegram] Dual-stack api.telegram.org path failed (%s)", failure)
                    continue
                logger.warning("[Telegram] IPv4 Telegram API IP %s failed: %s", ip, failure)
                await self._reset_fallback(ip)
                continue
        if last_error is None:
            raise RuntimeError("All Telegram fallback IPs exhausted but no error was recorded")
        raise last_error

    async def aclose(self) -> None:
        async with self._primary_lock:
            self._primary_closed = True
            primary = self._primary
        await primary.aclose()
        async with self._fallback_lock:
            transports = list(self._fallbacks.values())
            self._fallbacks.clear()
        for transport in transports:
            await transport.aclose()


def _normalize_fallback_ips(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        raw = str(value).strip()
        if not raw:
            continue
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            logger.warning("Ignoring invalid Telegram fallback IP: %r", raw)
            continue
        if addr.version != 4:
            logger.warning("Ignoring non-IPv4 Telegram fallback IP: %s", raw)
        elif addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_unspecified:
            logger.warning("Ignoring private/internal Telegram fallback IP: %s", raw)
        elif str(addr) in _MT_PROTO_ONLY_IPS:
            logger.warning("Ignoring MTProto-only Telegram DC (serves no Bot-API TLS): %s", raw)
        else:
            normalized.append(str(addr))
    return normalized


def _in_telegram_band(ip: str) -> bool:
    """True only when ``ip`` is an IPv4 literal inside a Telegram-owned network."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.version == 4 and any(addr in net for net in _TELEGRAM_NETWORKS)


# Fallback candidates are only trusted after an actual TCP + TLS handshake against the IP with
# SNI=api.telegram.org. The band filter (``_in_telegram_band``) is a cheap first pass, but it is not
# sufficient: Telegram's MTProto-only DCs live inside the same 149.154.160.0/20 band
# (e.g. 149.154.175.50) and accept a TCP connection on :443 while speaking MTProto, not Bot-API TLS
# — a fallback to such an IP raises SSL WRONG_VERSION on every request. The handshake result is the
# final judgement; failures are logged and dropped from the candidate list.
_TLS_PROBE_TIMEOUT = 5.0
_TLS_PROBE_PORT = 443


def _probe_tls(ip: str, timeout: float = _TLS_PROBE_TIMEOUT) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for a TCP connect + TLS handshake to ``ip`` with SNI=api.telegram.org."""
    context = ssl.create_default_context()
    try:
        with socket.create_connection((ip, _TLS_PROBE_PORT), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=_TELEGRAM_API_HOST):
                return True, ""
    except Exception as exc:
        return False, _describe_transport_error(exc)


def probe_fallback_ips(ips: Iterable[str], timeout: float = _TLS_PROBE_TIMEOUT) -> list[str]:
    """Validate fallback candidates with one TLS handshake each; return only the ones that complete it.

    Order is preserved. ``_in_telegram_band`` stays the first filter; the probe is the final judgement
    (it rejects MTProto-only DC IPs that sit inside the band but speak no Bot-API TLS). Called once
    when the candidate set is built / switches — never on the normal request path.
    """
    validated: list[str] = []
    for ip in dict.fromkeys(_normalize_fallback_ips(ips)):
        if not _in_telegram_band(ip):
            logger.warning("Discarding out-of-band Telegram fallback IP before TLS probe: %s", ip)
            continue
        ok, reason = _probe_tls(ip, timeout=timeout)
        if ok:
            validated.append(ip)
        else:
            logger.warning(
                "Telegram fallback IP %s failed TLS probe (SNI=%s) and was excluded: %s",
                ip, _TELEGRAM_API_HOST, reason)
    return validated


async def probe_fallback_ips_async(ips: Iterable[str], timeout: float = _TLS_PROBE_TIMEOUT) -> list[str]:
    """Thread-offloaded :func:`probe_fallback_ips` so the blocking handshake never stalls the loop."""
    return await asyncio.to_thread(probe_fallback_ips, list(ips), timeout)


def parse_fallback_ip_env(value: str | None) -> list[str]:
    return _normalize_fallback_ips(part.strip() for part in value.split(",")) if value else []


def _resolve_system_dns() -> set[str]:
    """Return the IPv4 addresses that the OS resolver gives for api.telegram.org."""
    try:
        results = socket.getaddrinfo(_TELEGRAM_API_HOST, 443, socket.AF_INET)
        return {str(addr[4][0]) for addr in results if _in_telegram_band(str(addr[4][0]))}
    except Exception:
        return set()


async def _query_doh_provider(client: httpx.AsyncClient, provider: dict) -> list[str]:
    """Query one DoH provider and return A-record IPs."""
    try:
        resp = await client.get(provider["url"], params=provider["params"], headers=provider["headers"])
        resp.raise_for_status()
        data = resp.json()
        ips: list[str] = []
        for answer in data.get("Answer", []):
            if answer.get("type") != 1:  # A record
                continue
            raw = answer.get("data", "").strip()
            try:
                ipaddress.ip_address(raw)
            except ValueError:
                continue
            ips.append(raw)
        return ips
    except Exception as exc:
        logger.debug("DoH query to %s failed: %s", provider["url"], exc)
        return []


async def discover_fallback_ips() -> list[str]:
    """Resolve api.telegram.org via Google + Cloudflare DoH; unique A records, in order. IPs matching the
    system resolver are deliberately KEPT (often the most reliable path). Falls back to
    ``SEED_FALLBACK_IPS`` only when DoH yields nothing usable.

    IPs that match the local system resolver are kept rather than excluded: in many networks the system-DNS
    IP is the most reliable path to api.telegram.org and a transient primary-path failure should be retried
    against the same address via the IP-rewrite path before the seed list is consulted (#14520).
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(_DOH_TIMEOUT)) as client:
        system_dns_task = asyncio.ensure_future(asyncio.to_thread(_resolve_system_dns))
        results = await asyncio.gather(*[_query_doh_provider(client, p) for p in _DOH_PROVIDERS], return_exceptions=True)
    # The getaddrinfo leg has no timeout of its own and only feeds the log line below — bound it.
    # The system-resolver leg runs socket.getaddrinfo in a worker thread with no timeout of its own — a
    # wedged OS resolver (broken VPN/DNS) can sit for minutes. Its result only feeds the no-usable-answers
    # log line below, so it must never gate discovery: bound it and move on (#63309). The DoH legs are
    # already bounded by the client timeout above.
    system_ips: set[str] = set()
    try:
        system_result = await asyncio.wait_for(system_dns_task, timeout=_DOH_TIMEOUT)
        if isinstance(system_result, set):
            system_ips = system_result
    except Exception:
        logger.debug("System-DNS resolution for %s did not complete in time", _TELEGRAM_API_HOST)
    doh_ips = [ip for r in results if isinstance(r, list) for ip in r]
    deduped = _normalize_fallback_ips(list(dict.fromkeys(doh_ips)))  # dedupe, keep order
    validated = [ip for ip in deduped if _in_telegram_band(ip)]
    for ip in deduped:
        if ip not in validated:
            logger.warning("Discarding out-of-band DoH answer for %s: %s", _TELEGRAM_API_HOST, ip)
    if validated:
        # Last-resort literals discovered as A records go behind the rest of the answers (stable order).
        validated.sort(key=lambda ip: ip in _LAST_RESORT_IPS)
        logger.debug("Discovered Telegram fallback IPs via DoH: %s", ", ".join(validated))
        return validated
    logger.info(
        "DoH discovery yielded no usable IPs (system DNS: %s); using seed fallback IPs %s",
        ", ".join(system_ips) or "unknown", ", ".join(SEED_FALLBACK_IPS))
    return list(SEED_FALLBACK_IPS)


def _rewrite_request_for_ip(request: httpx.Request, ip: str) -> httpx.Request:
    original_host = request.url.host or _TELEGRAM_API_HOST
    url = request.url.copy_with(host=ip)
    headers = request.headers.copy()
    headers["host"] = original_host
    extensions = dict(request.extensions)
    extensions["sni_hostname"] = original_host
    return httpx.Request(method=request.method, url=url, headers=headers, stream=request.stream, extensions=extensions)


def _is_retryable_connect_error(exc: Exception) -> bool:
    return isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError))
