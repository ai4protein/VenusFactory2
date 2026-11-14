"""RSI outputs are downloadable research cases, not conversational tools."""
RSI_TOOL_NAMES = frozenset({
    "list_evolved_models", "describe_evolved_model", "predict_with_evolved_model",
    "search_generated_capabilities", "describe_generated_capability", "run_generated_capability",
})
RSI_ARCHIVE_MESSAGE = "RSI results are available as case repositories in RSI Plaza, with previews and ZIP downloads. They are not callable tools."


def is_rsi_model(identity):
    return str(identity).startswith(("scientist:", "capability:"))
