"""OpenAI-compatible HTTP proxy server using FastAPI."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the proxy server. Install with: pip install tokenrouter[proxy]"
    )

from tokenrouter import TokenRouter
from tokenrouter.classifier import classify_async
from tokenrouter.config import TokenRouterConfig
from tokenrouter.fallback import chat_stream_with_fallback, chat_with_fallback
from tokenrouter.models import MODEL_REGISTRY, calculate_cost, get_model
from tokenrouter.types import ChatCompletionRequest, TokenRouterMetadata


def create_app(config_path: str | None = None, config: TokenRouterConfig | None = None) -> FastAPI:
    """Create the FastAPI proxy application."""
    if config_path:
        cfg = TokenRouterConfig.from_yaml(config_path)
    elif config:
        cfg = config
    else:
        cfg = TokenRouterConfig()

    router = TokenRouter(config=cfg)

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
            ],
        )

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

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request):
        start_time = time.time()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        body: dict[str, Any] = await request.json()

        if not body.get("messages"):
            return JSONResponse(
                {"error": {"message": "'messages' field is required", "type": "error", "code": 400}},
                status_code=400,
            )

        messages = body["messages"]
        strategy_header = request.headers.get("x-routing-strategy")
        strategy = strategy_header or body.get("strategy") or cfg.strategy
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
                keys=cfg.keys,
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

                    async for chunk, model_used, is_fallback, fallback_from in chat_stream_with_fallback(
                        req, model_config, fallback_chain, cfg.keys
                    ):
                        chunk.model = model_used.id
                        chunk.id = f"chatcmpl-{request_id}"
                        content = None
                        if chunk.choices:
                            content = chunk.choices[0].get("delta", {}).get("content")
                        if content:
                            total_content += content
                        yield f"data: {json.dumps(chunk.to_dict())}\n\n"

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
        result = await chat_with_fallback(req, model_config, fallback_chain, cfg.keys)
        response = result.response

        latency_ms = int((time.time() - start_time) * 1000)
        response.id = f"chatcmpl-{request_id}"
        response.model = result.model_used.id

        cost = calculate_cost(
            result.model_used,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        response._tokenrouter = TokenRouterMetadata(
            request_id=request_id,
            model_requested=body.get("model", "auto"),
            model_used=result.model_used.id,
            task_type=classification.task_type if classification else "explicit",
            complexity=classification.complexity if classification else "explicit",
            confidence=classification.confidence if classification else 1.0,
            cost=cost,
            latency_ms=latency_ms,
            strategy=strategy,
            fallback=result.is_fallback,
        )

        response_headers["X-TokenRouter-Cost"] = f"{cost:.6f}"

        return JSONResponse(response.to_dict(), headers=response_headers)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": "0.1.0"})

    return app
