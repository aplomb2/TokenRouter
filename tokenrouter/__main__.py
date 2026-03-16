"""CLI entry point — `tokenrouter serve` and `tokenrouter classify`."""

from __future__ import annotations

import argparse
import json
import sys


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the OpenAI-compatible proxy server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install tokenrouter[proxy]", file=sys.stderr)
        sys.exit(1)

    from tokenrouter.proxy import create_app

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_classify(args: argparse.Namespace) -> None:
    """Classify a prompt and show routing decision."""
    from tokenrouter import TokenRouter

    if args.config:
        router = TokenRouter.from_config(args.config)
    else:
        router = TokenRouter(strategy=args.strategy or "balanced")

    messages = [{"role": "user", "content": args.prompt}]
    result = router.classify(messages)

    output = {
        "task_type": result.task_type,
        "complexity": result.complexity,
        "model": result.selected_model.id,
        "model_name": result.selected_model.name,
        "provider": result.selected_model.provider,
        "confidence": round(result.confidence, 3),
        "reasoning": result.reasoning,
        "fallback_chain": result.fallback_chain,
        "classifier": result.classifier_used,
    }
    if result.code_language:
        output["code_language"] = result.code_language
    if result.user_language:
        output["user_language"] = result.user_language

    print(json.dumps(output, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tokenrouter",
        description="TokenRouter — Intelligent AI model routing SDK",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible proxy server")
    serve_parser.add_argument("--config", "-c", help="Path to config.yaml")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")

    # classify
    classify_parser = subparsers.add_parser("classify", help="Classify a prompt and show routing")
    classify_parser.add_argument("prompt", help="The prompt to classify")
    classify_parser.add_argument("--config", "-c", help="Path to config.yaml")
    classify_parser.add_argument("--strategy", "-s", choices=["cheapest", "best", "balanced"])

    args = parser.parse_args()
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "classify":
        cmd_classify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
