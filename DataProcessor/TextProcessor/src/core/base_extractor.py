from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseExtractor(ABC):
    """
    Базовый интерфейс для всех экстракторов признаков.
    Экстрактор принимает входной документ и возвращает словарь признаков.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def extract(self, doc: Any) -> Dict[str, Any]:
        """
        Выполнить извлечение признаков из входного документа.

        :param doc: входной объект документа (см. schemas.models.VideoDocument)
        :return: словарь признаков {feature_name: value}
        """
        raise NotImplementedError

    def extract_batch(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """
        Обработка нескольких документов батчем.

        По умолчанию: последовательный вызов `extract()` для каждого документа (обратная совместимость).
        Extractor'ы могут переопределить этот метод для GPU batching / CPU parallel friendly реализации.
        """
        return [self.extract(d) for d in (docs or [])]

    @property
    def supports_batch(self) -> bool:
        """
        Возвращает True, если extractor реализует оптимизированный `extract_batch()`.

        Важно: наличие метода `extract_batch()` есть у всех (дефолтная реализация),
        поэтому этот флаг позволяет отличать оптимизированную батчевую реализацию от fallback.
        """
        return False
