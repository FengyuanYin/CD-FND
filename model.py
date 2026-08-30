from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo


def get_model(config, role: str):
    normalized_role = role.casefold()
    if normalized_role not in {"analysis", "coordinate", "optimization", "judge"}:
        raise ValueError(f"Unsupported model role: {role}")

    return OpenAIChatCompletionClient(
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.base_url,
        parallel_tool_calls=False,
        model_info=ModelInfo(
            vision=getattr(config, "vision_support", False),
            function_calling=getattr(config, "function_calling", True),
            json_output=getattr(config, "json_output", True),
            family="unknown",
            structured_output=getattr(config, "structured_output", True),
        ),
    )
