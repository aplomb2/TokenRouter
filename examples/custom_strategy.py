"""Custom routing strategy and rules example."""

from tokenrouter import TokenRouter
from tokenrouter.types import CustomRule

# Force specific models for specific tasks
router = TokenRouter(
    keys={
        "openai": "sk-...",
        "deepseek": "sk-...",
        "qwen": "sk-...",
    },
    strategy="cheapest",
    rules=[
        CustomRule(task="coding", model="deepseek-chat"),       # Always use DeepSeek for coding
        CustomRule(task="chinese_language", model="qwen-max"),   # Chinese tasks -> Qwen
    ],
    exclude_models=["claude-opus-4"],  # Too expensive
)

# This will use DeepSeek because of the custom rule
response = router.chat([
    {"role": "user", "content": "Implement a binary search tree in Python"}
])
print(f"Model: {response.model_used}")

# This will use cheapest available model (no custom rule for simple_qa)
response = router.chat([
    {"role": "user", "content": "What is the capital of France?"}
])
print(f"Model: {response.model_used}")

# Override strategy per-request
response = router.chat(
    [{"role": "user", "content": "Explain quantum computing"}],
    strategy="best",  # Use best model for this one request
)
print(f"Model: {response.model_used}")
