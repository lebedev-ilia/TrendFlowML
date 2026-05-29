"""File-first dataset collector for large training crawls."""

from fetcher.dataset_collector.config import CampaignConfig, load_campaign_config
from fetcher.dataset_collector.state import DatasetState

__all__ = ["CampaignConfig", "DatasetState", "load_campaign_config"]
