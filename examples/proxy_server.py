"""Start the OpenAI-compatible proxy server programmatically."""

import uvicorn

from tokenrouter.config import TokenRouterConfig, ProxyConfig
from tokenrouter.proxy import create_app
from tokenrouter.types import CustomRule

# Configure programmatically (or use TokenRouterConfig.from_yaml("config.yaml"))
config = TokenRouterConfig(
    keys={
        "openai": "sk-...",
        "anthropic": "sk-ant-...",
        "deepseek": "sk-...",
    },
    strategy="balanced",
    rules=[
        CustomRule(task="coding", model="deepseek-chat"),
    ],
    proxy=ProxyConfig(host="0.0.0.0", port=8000, cors=True),
)

app = create_app(config=config)

if __name__ == "__main__":
    print("Starting TokenRouter proxy on http://0.0.0.0:8000")
    print("Use with any OpenAI SDK:")
    print('  client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")')
    uvicorn.run(app, host=config.proxy.host, port=config.proxy.port)
