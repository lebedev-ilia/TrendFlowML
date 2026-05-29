from fetcher.dataset_collector.discovery.base import DiscoveryAdapter, DiscoveryCapabilities
from fetcher.dataset_collector.discovery.rutube import RutubeDiscoveryAdapter
from fetcher.dataset_collector.discovery.tiktok import TikTokDiscoveryAdapter
from fetcher.dataset_collector.discovery.twitch import TwitchDiscoveryAdapter
from fetcher.dataset_collector.discovery.youtube import YouTubeDiscoveryAdapter, YouTubeKeyPool

__all__ = [
    "DiscoveryAdapter",
    "DiscoveryCapabilities",
    "RutubeDiscoveryAdapter",
    "TikTokDiscoveryAdapter",
    "TwitchDiscoveryAdapter",
    "YouTubeDiscoveryAdapter",
    "YouTubeKeyPool",
]
