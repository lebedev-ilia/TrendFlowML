"""
Security модуль для аутентификации API

Реализует:
- API Key аутентификацию (MVP)
- Инфраструктуру для mTLS (будущее)

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2038-2091)
"""

import os
import logging
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from api.config import config

logger = logging.getLogger(__name__)

# API Key Header
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API Key для аутентификации"
)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    Проверка API Key для аутентификации.
    
    Args:
        api_key: API Key из заголовка X-API-Key
        
    Returns:
        API Key если валиден
        
    Raises:
        HTTPException 401: Если API key не предоставлен
        HTTPException 403: Если API key невалиден
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2048-2065)
    """
    # Проверка что API key предоставлен
    if not api_key:
        logger.warning("API key not provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Получить валидный API key из конфигурации
    valid_key = config.api_key
    
    # Если API key не настроен, разрешить доступ (для development)
    if not valid_key:
        logger.warning("API key not configured, allowing access (development mode)")
        return api_key
    
    # Проверка соответствия API key
    if api_key != valid_key:
        logger.warning(f"Invalid API key provided: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    logger.debug("API key verified successfully")
    return api_key


# ============================================================================
# mTLS инфраструктура (будущее)
# ============================================================================

async def verify_mtls(request) -> Optional[str]:
    """
    Проверка mTLS сертификата (будущее).
    
    Этот метод подготовлен для будущей реализации mTLS аутентификации.
    В текущей версии не используется.
    
    Args:
        request: FastAPI Request объект
        
    Returns:
        CN (Common Name) из сертификата если валиден
        
    Raises:
        HTTPException 401: Если клиентский сертификат не предоставлен или невалиден
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2075-2091)
    """
    # TODO: Реализовать проверку клиентского сертификата
    # Для production нужно:
    # 1. Проверить наличие клиентского сертификата в request.client.cert
    # 2. Загрузить сертификат через cryptography.x509
    # 3. Проверить CN, issuer, expiration
    # 4. Проверить цепочку сертификатов
    # 5. Проверить отзыв сертификата (OCSP/CRL)
    
    client_cert = getattr(request.client, "cert", None)
    if not client_cert:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Client certificate required",
        )
    
    # TODO: Реализовать валидацию сертификата
    # from cryptography import x509
    # cert = x509.load_pem_x509_certificate(client_cert)
    # cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
    # return cn
    
    raise NotImplementedError("mTLS authentication not yet implemented")


def get_auth_dependency():
    """
    Получить dependency для аутентификации.
    
    Возвращает соответствующую функцию аутентификации в зависимости от конфигурации.
    По умолчанию использует API Key аутентификацию.
    
    Returns:
        Функция аутентификации (verify_api_key или verify_mtls)
    """
    # В будущем можно добавить логику выбора между API Key и mTLS
    # на основе конфигурации
    auth_type = getattr(config, "auth_type", "api_key")
    
    if auth_type == "mtls":
        # Для mTLS нужен доступ к Request объекту
        # Это будет реализовано через middleware или dependency с Request
        logger.warning("mTLS authentication not yet implemented, using API Key")
        return verify_api_key
    
    return verify_api_key

