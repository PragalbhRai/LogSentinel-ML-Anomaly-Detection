"""The three destinations in the faculty review dashboard."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageSpec:
    key: str
    title: str
    description: str


PAGES = (
    PageSpec("overview", "Overview", "Understanding the data behind LogSentinel."),
    PageSpec("model-comparison", "Model Comparison", "Three models. The same saved test set."),
    PageSpec("anomaly-detection", "Anomaly Detection", "See how each model classified a saved test block."),
)
