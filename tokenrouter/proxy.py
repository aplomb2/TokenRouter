"""OpenAI-compatible HTTP proxy server with unified key management and billing."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the proxy server. Install with: pip install tokenrouter[proxy]"
    )

from tokenrouter import TokenRouter
from tokenrouter.billing import BillingEngine, calculate_baseline_cost
from tokenrouter.classifier import classify_async
from tokenrouter.config import TokenRouterConfig
from tokenrouter.fallback import chat_stream_with_fallback, chat_with_fallback
from tokenrouter.keys import AsyncKeyStore, KeyStore, TRKey
from tokenrouter.models import MODEL_REGISTRY, calculate_cost, get_model
from tokenrouter.types import ChatCompletionRequest, TokenRouterMetadata

_DASHBOARD_HTML: str | None = None


def _load_dashboard_html() -> str:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        html_path = Path(__file__).parent / "dashboard" / "index.html"
        _DASHBOARD_HTML = html_path.read_text(encoding="utf-8")
    return _DASHBOARD_HTML


def _has_tr_keys(cfg: TokenRouterConfig) -> bool:
    """Check if the system is in multi-tenant key mode (has admin_token or database configured)."""
    return bool(cfg.admin_token or cfg.database)


def create_app(config_path: str | None = None, config: TokenRouterConfig | None = None) -> FastAPI:
    """Create the FastAPI proxy application."""
    if config_path:
        cfg = TokenRouterConfig.from_yaml(config_path)
    elif config:
        cfg = config
    else:
        cfg = TokenRouterConfig()

    router = TokenRouter(config=cfg)

    # Resolve admin token from env if not in config
    admin_token = cfg.admin_token or os.environ.get("TOKENROUTER_ADMIN_TOKEN", "")

    # Key store (if multi-tenant mode)
    db_path = cfg.database or os.path.join(Path.home(), ".tokenrouter", "keys.db")
    key_store = AsyncKeyStore(db_path)

    # Billing engine
    billing: BillingEngine | None = None

    def _get_billing() -> BillingEngine:
        nonlocal billing
        if billing is None:
            billing = BillingEngine(key_store._get_store()._conn)
        return billing

    app = FastAPI(title="TokenRouter Proxy", version="0.1.0")

    if cfg.proxy.cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-TokenRouter-Model",
                "X-TokenRouter-Task",
                "X-TokenRouter-Complexity",
                "X-TokenRouter-Request-ID",
                "X-TokenRouter-Cost",
                "X-TokenRouter-Saved",
                "X-TokenRouter-Saved-Pct",
            ],
        )

    # === Helper: auth ===

    def _extract_bearer(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    def _require_admin(request: Request) -> bool:
        """Check if request has valid admin token."""
        if not admin_token:
            return True  # No admin token configured = open access
        token = _extract_bearer(request)
        return token == admin_token

    async def _resolve_keys(request: Request) -> tuple[dict[str, str], TRKey | None, str | None]:
        """Resolve provider keys from request auth.

        Returns (keys_dict, tr_key_or_none, error_message_or_none).
        In legacy mode (direct keys in config), returns config keys.
        In multi-tenant mode, validates tr-xxx key and returns its providers.
        """
        token = _extract_bearer(request)

        # If token is a tr-xxx key, validate it
        if token.startswith("tr-"):
            tr_key = await key_store.validate_key(token)
            if not tr_key:
                return {}, None, "Invalid API key"
            if not tr_key.providers:
                return {}, None, "No provider keys configured for this API key"
            return tr_key.providers, tr_key, None

        # Legacy mode: use config keys if available
        if cfg.keys:
            return cfg.keys, None, None

        # No keys at all
        return {}, None, "No API key provided. Use Authorization: Bearer tr-xxx"

    # === Dashboard ===

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(_load_dashboard_html())

    # === Dashboard API endpoints ===

    @app.get("/v1/dashboard/stats")
    async def dashboard_stats(request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return JSONResponse(_get_billing().get_global_stats())

    @app.get("/v1/dashboard/daily")
    async def dashboard_daily(request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        costs = _get_billing().get_global_daily_costs()
        return JSONResponse([{"date": c.date, "actual_cost": c.actual_cost, "baseline_cost": c.baseline_cost} for c in costs])

    @app.get("/v1/dashboard/models")
    async def dashboard_models(request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return JSONResponse(_get_billing().get_global_model_distribution())

    @app.get("/v1/dashboard/logs")
    async def dashboard_logs(request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return JSONResponse(_get_billing().get_global_recent_logs())

    # === Key Management API ===

    @app.post("/v1/keys")
    async def create_key(request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        body: dict[str, Any] = await request.json()
        name = body.get("name", "unnamed")
        strategy = body.get("strategy", "balanced")
        rate_limit = body.get("rate_limit", 0)
        tr_key = await key_store.create_key(name, strategy, rate_limit)
        return JSONResponse(tr_key.to_dict(mask_keys=False), status_code=201)

    @app.get("/v1/keys")
    async def list_keys(request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        keys = await key_store.list_keys()
        return JSONResponse({"keys": [k.to_dict() for k in keys]})

    @app.delete("/v1/keys/{key_id}")
    async def delete_key(key_id: int, request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        ok = await key_store.delete_key(key_id)
        if not ok:
            return JSONResponse({"error": "Key not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    @app.post("/v1/keys/{key_id}/providers")
    async def add_provider(key_id: int, request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        body: dict[str, Any] = await request.json()
        provider = body.get("provider", "")
        api_key = body.get("api_key", "")
        if not provider or not api_key:
            return JSONResponse({"error": "provider and api_key are required"}, status_code=400)
        ok = await key_store.add_provider(key_id, provider, api_key)
        if not ok:
            return JSONResponse({"error": "Failed to add provider"}, status_code=400)
        return JSONResponse({"status": "added"}, status_code=201)

    @app.delete("/v1/keys/{key_id}/providers/{provider}")
    async def remove_provider(key_id: int, provider: str, request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        ok = await key_store.remove_provider(key_id, provider)
        if not ok:
            return JSONResponse({"error": "Provider not found"}, status_code=404)
        return JSONResponse({"status": "removed"})

    # === Usage API ===

    @app.get("/v1/usage/{key_id}")
    async def get_usage(key_id: int, request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        period = request.query_params.get("period", "all")
        summary = _get_billing().get_usage_summary(key_id, period)
        return JSONResponse(summary.to_dict())

    @app.get("/v1/usage/{key_id}/savings")
    async def get_savings(key_id: int, request: Request) -> JSONResponse:
        if not _require_admin(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        report = _get_billing().get_savings_report(key_id)
        return JSONResponse(report.to_dict())

    # === Model listing ===

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        models = [
            {
                "id": m.id,
                "object": "model",
                "owned_by": m.provider,
                "created": 0,
            }
            for m in router.available_models
        ]
        return JSONResponse({"object": "list", "data": models})

    # === Chat Completions ===

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request):
        start_time = time.time()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        # Auth & key resolution
        keys, tr_key, auth_error = await _resolve_keys(request)
        if auth_error:
            return JSONResponse(
                {"error": {"message": auth_error, "type": "auth_error", "code": 401}},
                status_code=401,
            )

        body: dict[str, Any] = await request.json()

        if not body.get("messages"):
            return JSONResponse(
                {"error": {"message": "'messages' field is required", "type": "error", "code": 400}},
                status_code=400,
            )

        messages = body["messages"]
        strategy_header = request.headers.get("x-routing-strategy")
        # Use tr_key strategy if available, then header, then body, then config
        strategy = strategy_header or body.get("strategy") or (tr_key.strategy if tr_key else None) or cfg.strategy
        stream = body.get("stream", False)

        req = ChatCompletionRequest(
            messages=messages,
            model=body.get("model", "auto"),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            top_p=body.get("top_p"),
            stream=stream,
            stop=body.get("stop"),
            presence_penalty=body.get("presence_penalty"),
            frequency_penalty=body.get("frequency_penalty"),
            user=body.get("user"),
        )

        # Determine model
        explicit_model = body.get("model")
        if explicit_model and explicit_model != "auto":
            model_config = get_model(explicit_model)
            if not model_config:
                # Try by provider_model_id
                model_config = next(
                    (m for m in MODEL_REGISTRY if m.provider_model_id == explicit_model), None
                )
            if not model_config:
                return JSONResponse(
                    {"error": {"message": f"Model '{explicit_model}' not found.", "type": "error", "code": 400}},
                    status_code=400,
                )
            classification = None
            fallback_chain: list[str] = []
        else:
            classification = await classify_async(
                messages,
                strategy=strategy,
                custom_rules=cfg.rules or None,
                keys=keys,
                exclude_models=cfg.exclude_models or None,
            )
            model_config = classification.selected_model
            fallback_chain = classification.fallback_chain

        response_headers = {
            "X-TokenRouter-Model": model_config.id,
            "X-TokenRouter-Task": classification.task_type if classification else "explicit",
            "X-TokenRouter-Complexity": classification.complexity if classification else "explicit",
            "X-TokenRouter-Request-ID": request_id,
        }

        if stream:
            async def generate():
                total_content = ""
                try:
                    # Initial chunk with role
                    init_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_config.id,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(init_chunk)}\n\n"

                    final_model_used = model_config
                    async for chunk, model_used, is_fallback, fallback_from in chat_stream_with_fallback(
                        req, model_config, fallback_chain, keys
                    ):
                        final_model_used = model_used
                        chunk.model = model_used.id
                        chunk.id = f"chatcmpl-{request_id}"
                        content = None
                        if chunk.choices:
                            content = chunk.choices[0].get("delta", {}).get("content")
                        if content:
                            total_content += content
                        yield f"data: {json.dumps(chunk.to_dict())}\n\n"

                    # Estimate tokens for streaming (rough: 1 token ≈ 4 chars)
                    est_input = sum(len(str(m.get("content", ""))) for m in messages) // 4
                    est_output = len(total_content) // 4
                    cost = calculate_cost(final_model_used, est_input, est_output)
                    baseline = calculate_baseline_cost(est_input, est_output, cfg.baseline_model)
                    saved = baseline - cost

                    # Final chunk with cost info
                    final_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": final_model_used.id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "_tokenrouter": {
                            "cost": round(cost, 6),
                            "baseline_cost": round(baseline, 6),
                            "saved": round(saved, 6),
                            "saved_pct": round(saved / baseline * 100, 1) if baseline > 0 else 0.0,
                            "model_used": final_model_used.id,
                        },
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

                    # Log to billing
                    if tr_key:
                        await key_store.log_request(
                            tr_key_id=tr_key.id,
                            model_requested=body.get("model", "auto"),
                            model_used=final_model_used.id,
                            task_type=classification.task_type if classification else "explicit",
                            complexity=classification.complexity if classification else "explicit",
                            input_tokens=est_input,
                            output_tokens=est_output,
                            actual_cost=cost,
                            baseline_cost=baseline,
                            latency_ms=int((time.time() - start_time) * 1000),
                            success=True,
                        )

                    yield "data: [DONE]\n\n"
                except Exception as err:
                    error_data = {"error": {"message": str(err)}}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={**response_headers, "Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming
        result = await chat_with_fallback(req, model_config, fallback_chain, keys)
        response = result.response

        latency_ms = int((time.time() - start_time) * 1000)
        response.id = f"chatcmpl-{request_id}"
        response.model = result.model_used.id

        cost = calculate_cost(
            result.model_used,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        baseline = calculate_baseline_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            cfg.baseline_model,
        )
        saved = baseline - cost
        saved_pct = (saved / baseline * 100) if baseline > 0 else 0.0

        response._tokenrouter = TokenRouterMetadata(
            request_id=request_id,
            model_requested=body.get("model", "auto"),
            model_used=result.model_used.id,
            task_type=classification.task_type if classification else "explicit",
            complexity=classification.complexity if classification else "explicit",
            confidence=classification.confidence if classification else 1.0,
            cost=cost,
            cost_if_opus=baseline,
            latency_ms=latency_ms,
            strategy=strategy,
            fallback=result.is_fallback,
        )

        response_headers["X-TokenRouter-Cost"] = f"{cost:.6f}"
        response_headers["X-TokenRouter-Saved"] = f"{saved:.6f}"
        response_headers["X-TokenRouter-Saved-Pct"] = f"{saved_pct:.1f}"

        # Log to billing
        if tr_key:
            await key_store.log_request(
                tr_key_id=tr_key.id,
                model_requested=body.get("model", "auto"),
                model_used=result.model_used.id,
                task_type=classification.task_type if classification else "explicit",
                complexity=classification.complexity if classification else "explicit",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                actual_cost=cost,
                baseline_cost=baseline,
                latency_ms=latency_ms,
                success=True,
            )

        return JSONResponse(response.to_dict(), headers=response_headers)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": "0.1.0"})

    return app
