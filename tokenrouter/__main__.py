"""CLI entry point — `tokenrouter serve`, `tokenrouter classify`, `tokenrouter keys`, `tokenrouter usage`."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


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


def cmd_keys(args: argparse.Namespace) -> None:
    """Key management commands."""
    from tokenrouter.keys import KeyStore

    db_path = args.database or ""
    if db_path:
        store = KeyStore(db_path)
    else:
        store = KeyStore()

    try:
        if args.keys_command == "create":
            key = store.create_key(name=args.name, strategy=args.strategy or "balanced")
            print(f"Created key: {key.api_key}")
            print(f"  ID: {key.id}")
            print(f"  Name: {key.name}")
            print(f"  Strategy: {key.strategy}")

        elif args.keys_command == "list":
            keys = store.list_keys()
            if not keys:
                print("No keys found.")
                return
            for k in keys:
                created = datetime.fromtimestamp(k.created_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                providers = ", ".join(k.providers.keys()) if k.providers else "none"
                print(f"  [{k.id}] {k.name} | {k.api_key} | strategy={k.strategy} | providers={providers} | {created}")

        elif args.keys_command == "add-provider":
            key_id = int(args.key_id)
            ok = store.add_provider(key_id, args.provider, args.api_key)
            if ok:
                print(f"Added {args.provider} to key {key_id}")
            else:
                print("Failed to add provider.", file=sys.stderr)
                sys.exit(1)

        elif args.keys_command == "delete":
            key_id = int(args.key_id)
            ok = store.delete_key(key_id)
            if ok:
                print(f"Deleted key {key_id}")
            else:
                print("Key not found.", file=sys.stderr)
                sys.exit(1)

        else:
            print("Unknown keys command. Use: create, list, add-provider, delete")
    finally:
        store.close()


def cmd_usage(args: argparse.Namespace) -> None:
    """Show usage and savings for a key."""
    from tokenrouter.billing import BillingEngine
    from tokenrouter.keys import KeyStore

    db_path = args.database or ""
    store = KeyStore(db_path) if db_path else KeyStore()

    try:
        billing = BillingEngine(store._conn)
        key_id = int(args.key_id)
        period = args.period or "all"

        summary = billing.get_usage_summary(key_id, period)
        report = billing.get_savings_report(key_id)

        print(f"Usage for key {key_id} (period: {period})")
        print(f"  Total requests:  {summary.total_requests}")
        print(f"  Successful:      {summary.successful_requests}")
        print(f"  Input tokens:    {summary.total_input_tokens:,}")
        print(f"  Output tokens:   {summary.total_output_tokens:,}")
        print(f"  Total cost:      ${summary.total_cost:.6f}")
        print(f"  Baseline cost:   ${summary.total_baseline_cost:.6f}")
        print(f"  Saved:           ${summary.total_saved:.6f} ({summary.saved_pct:.1f}%)")
        print()
        print("Savings Report (all time):")
        print(f"  Total cost:      ${report.total_cost:.6f}")
        print(f"  Baseline cost:   ${report.baseline_cost:.6f}")
        print(f"  Saved:           ${report.saved:.6f} ({report.saved_pct:.1f}%)")
        print(f"  Request count:   {report.request_count}")

        if summary.models_used:
            print()
            print("Models used:")
            for model, count in summary.models_used.items():
                print(f"  {model}: {count}")
    finally:
        store.close()


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

    # keys
    keys_parser = subparsers.add_parser("keys", help="Manage API keys")
    keys_parser.add_argument("--database", "-d", help="Path to keys.db")
    keys_sub = keys_parser.add_subparsers(dest="keys_command")

    keys_create = keys_sub.add_parser("create", help="Create a new API key")
    keys_create.add_argument("--name", "-n", required=True, help="Key name")
    keys_create.add_argument("--strategy", "-s", choices=["cheapest", "best", "balanced"], default="balanced")

    keys_sub.add_parser("list", help="List all API keys")

    keys_add_prov = keys_sub.add_parser("add-provider", help="Add a provider key to a TR key")
    keys_add_prov.add_argument("key_id", help="TR key ID")
    keys_add_prov.add_argument("provider", help="Provider name (openai, anthropic, ...)")
    keys_add_prov.add_argument("api_key", help="Provider API key")

    keys_delete = keys_sub.add_parser("delete", help="Delete an API key")
    keys_delete.add_argument("key_id", help="TR key ID")

    # usage
    usage_parser = subparsers.add_parser("usage", help="Show usage and savings for a key")
    usage_parser.add_argument("key_id", help="TR key ID")
    usage_parser.add_argument("--period", choices=["today", "week", "month", "all"], default="all")
    usage_parser.add_argument("--database", "-d", help="Path to keys.db")

    args = parser.parse_args()
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "keys":
        if not hasattr(args, "keys_command") or not args.keys_command:
            keys_parser.print_help()
        else:
            cmd_keys(args)
    elif args.command == "usage":
        cmd_usage(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
