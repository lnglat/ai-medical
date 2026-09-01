from langchain_deepseek import ChatDeepSeek

from src.config.settings import settings


def get_chat_model(temperature: float = 0):
    """创建唯一允许使用的 DeepSeek 模型实例。

    模型名称和 API Key 均来自 .env；这里不保留第二家模型的回退分支，避免
    配置来源分散或在日志、代码中误存额外密钥。
    """
    if settings.llm_provider.lower() != "deepseek":
        raise ValueError("LLM_PROVIDER must be 'deepseek'.")
    if settings.deepseek_api_key is None or not settings.deepseek_api_key.get_secret_value():
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，无法启用真实 LLM")
    return ChatDeepSeek(
        model=settings.deepseek_model,
        temperature=temperature,
        api_key=settings.deepseek_api_key.get_secret_value(),
    )
