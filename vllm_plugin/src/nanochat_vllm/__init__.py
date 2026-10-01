"""vLLM registration entry point for Nanochat."""


def register() -> None:
    """Register the out-of-tree model without importing CUDA eagerly."""
    from vllm import ModelRegistry

    architecture = "NanochatForCausalLM"
    if architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            architecture,
            "nanochat_vllm.model:NanochatForCausalLM",
        )


__all__ = ["register"]

