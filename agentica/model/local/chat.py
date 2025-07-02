from typing import Optional
from os import getenv

from agentica.model.openai.like import OpenAILike


class LocalChat(OpenAILike):
    """
    A model class for DeepSeek Chat API.

    Attributes:
    - id: str: The unique identifier of the model. Default: "deepseek-chat".
    - name: str: The name of the model. Default: "DeepSeekChat".
    - provider: str: The provider of the model. Default: "DeepSeek".
    - api_key: Optional[str]: The API key for the model.
    - base_url: str: The base URL for the model.
    """

    id: str = "qwen3-weixin"
    name: str = "qwen3-weixin"
    provider: str = "qwen3-weixin"

    api_key: Optional[str] = getenv("LOCAL_API_KEY", None)
    base_url: str = "http://27.159.93.62:8087/v1"
    extra_body: dict = dict(top_k=20,chat_template_keargs=dict(enable_thinking=False))
