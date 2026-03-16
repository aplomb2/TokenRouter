"""Basic usage example — auto-route to optimal model based on prompt content."""

from tokenrouter import TokenRouter

router = TokenRouter(
    keys={
        "openai": "sk-...",
        "anthropic": "sk-ant-...",
        "deepseek": "sk-...",
    },
    strategy="balanced",
)

# Auto-route: TokenRouter picks the best model for the task
response = router.chat([
    {"role": "user", "content": "Write a Python function to sort a list using quicksort"}
])

print(f"Model used: {response.model_used}")
print(f"Response: {response.choices[0].message['content']}")

# Classify without calling any API (L1 only)
result = router.classify([
    {"role": "user", "content": "Translate this to French: Hello world"}
])
print(f"\nClassification: {result.task_type} ({result.complexity})")
print(f"Would route to: {result.selected_model.name}")
print(f"Confidence: {result.confidence:.2f}")
