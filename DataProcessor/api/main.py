"""
FastAPI приложение для DataProcessor API

Этот модуль содержит основное FastAPI приложение и точку входа для API сервера.

Основные компоненты:
- FastAPI app с настройками CORS и middleware
- Подключение роутеров из endpoints/
- Обработка ошибок
- Логирование

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1362)
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import config
from api.middleware.request_id import RequestIDMiddleware

# OpenTelemetry tracing (опционально)
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    
    # Экспортеры (выбираются через переменные окружения)
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        OTLP_AVAILABLE = True
    except ImportError:
        OTLP_AVAILABLE = False
    
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        JAEGER_AVAILABLE = True
    except ImportError:
        JAEGER_AVAILABLE = False
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    FastAPIInstrumentor = None

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    Limiter = None
    RateLimitExceeded = None
    get_remote_address = None

# Инициализация limiter (будет создан после определения get_backend_id)
limiter = None


def get_backend_id(request: Request) -> str:
    """
    Получить идентификатор backend instance для rate limiting.
    
    Использует заголовок X-Backend-ID если он предоставлен,
    иначе fallback на IP адрес клиента.
    
    Args:
        request: FastAPI Request объект
        
    Returns:
        Строка с идентификатором backend (X-Backend-ID или IP адрес)
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2787-2796)
    """
    # Попробовать получить X-Backend-ID из заголовков
    backend_id = request.headers.get("X-Backend-ID")
    if backend_id:
        return backend_id
    
    # Fallback на IP адрес клиента
    if get_remote_address:
        return get_remote_address(request)
    
    # Если slowapi не доступен, использовать client.host
    if request.client:
        return request.client.host
    
    return "unknown"
from api.utils.errors import (
    RunNotFoundError,
    InvalidPayloadError,
    ProcessingError,
    RunAlreadyExistsError,
    BackpressureError
)

# Настройка логирования
def setup_logging():
    """
    Настройка логирования для API сервера.
    
    Поддерживает два формата:
    - json: для production (структурированные логи)
    - text: для development (читаемый формат)
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2524-2544)
    """
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    if config.log_format == "json":
        # JSON формат для production
        try:
            from pythonjsonlogger import jsonlogger
            
            log_handler = logging.StreamHandler()
            # Используем JsonFormatter с поддержкой дополнительных полей
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d",
                timestamp=True
            )
            log_handler.setFormatter(formatter)
            
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            
            # Удаляем существующие handlers чтобы избежать дублирования
            root_logger.handlers = []
            root_logger.addHandler(log_handler)
            
            # Предотвратить распространение на root logger
            root_logger.propagate = False
            
        except ImportError:
            # Fallback на обычный формат если python-json-logger не установлен
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
    else:
        # Текстовый формат для development
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


# Инициализация логирования
setup_logging()
logger = logging.getLogger(__name__)


# Глобальная переменная для отслеживания времени запуска
_startup_time: Optional[float] = None

# Глобальная переменная для graceful shutdown
_shutdown_event: Optional[asyncio.Event] = None

def get_startup_time() -> Optional[float]:
    """Получить время запуска приложения."""
    return _startup_time

def get_uptime_seconds() -> float:
    """Получить время работы приложения в секундах."""
    global _startup_time
    if _startup_time is None:
        return 0.0
    return time.time() - _startup_time

def get_shutdown_event() -> Optional[asyncio.Event]:
    """Получить shutdown event для graceful shutdown."""
    return _shutdown_event

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    
    Startup: инициализация ресурсов
    Shutdown: graceful shutdown с ожиданием завершения активных запросов
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2287-2317)
    """
    global _startup_time, _shutdown_event
    
    # Startup
    _startup_time = time.time()
    _shutdown_event = asyncio.Event()
    
    logger.info("Starting DataProcessor API server...")
    logger.info(f"API will be available at http://{config.api_host}:{config.api_port}")
    logger.info(f"Max concurrent runs: {config.max_concurrent_runs}")
    
    # Валидация конфигурации перед инициализацией зависимостей
    try:
        from api.utils.config_validator import validate_config, ConfigValidationError
        validate_config()
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed, aborting startup: {e}")
        # Пробросить исключение, чтобы приложение не стартовало с невалидной конфигурацией
        raise
    except Exception as e:
        logger.error(f"Unexpected error during configuration validation: {e}")
        raise
    
    # Инициализация Redis (опционально, для Этапа 2)
    try:
        from api.services.redis_client import init_redis_client
        redis_client = await init_redis_client()
        if redis_client:
            logger.info("Redis client initialized successfully")
        else:
            logger.info("Redis not configured, continuing without Redis")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis (continuing without Redis): {e}")
    
    # Инициализация OpenTelemetry tracing будет выполнена после создания app
    # (см. код после app.include_router)
    
    yield
    
    # Graceful Shutdown
    logger.info("Shutting down DataProcessor API server...")
    
    # 1. Stop accepting new requests
    # FastAPI автоматически перестанет принимать новые запросы при выходе из lifespan
    _shutdown_event.set()
    logger.info("Stopped accepting new requests")
    
    # 2. Wait for current requests
    # Получить TaskManager для проверки активных run'ов
    try:
        from api.services.task_manager import TaskManager
        from api.dependencies import get_task_manager
        task_manager = get_task_manager()
        
        active_runs = task_manager.get_active_runs_count() if task_manager else 0
        if active_runs > 0:
            logger.info(f"Waiting for {active_runs} active runs to complete...")
            # Подождать немного для завершения активных запросов
            # В production можно добавить более сложную логику ожидания
            await asyncio.sleep(5)  # Дать время на завершение текущих запросов
            logger.info("Finished waiting for active runs")
    except Exception as e:
        logger.warning(f"Error waiting for active runs: {e}")
    
    # 3. Cleanup resources
    # Закрытие Redis подключения
    try:
        from api.services.redis_client import close_redis_client
        await close_redis_client()
        logger.info("Redis client closed")
    except Exception as e:
        logger.warning(f"Error closing Redis client: {e}")
    
    _startup_time = None
    _shutdown_event = None
    logger.info("API server shutdown complete")


def _setup_opentelemetry(app_instance: FastAPI):
    """
    Настройка OpenTelemetry tracing для FastAPI приложения.
    
    Поддерживает два экспортера:
    - Jaeger (через UDP agent)
    - OTLP (через gRPC endpoint)
    
    Args:
        app_instance: FastAPI приложение для инструментации
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2546-2562)
    """
    if not OPENTELEMETRY_AVAILABLE:
        return
    
    # Создать Resource с метаданными сервиса
    resource = Resource.create({
        "service.name": config.service_name,
        "service.version": config.service_version,
        "service.namespace": "dataprocessor",
    })
    
    # Настроить TracerProvider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    # Выбрать экспортер на основе конфигурации
    exporter = None
    if config.tracing_exporter == "jaeger" and JAEGER_AVAILABLE:
        exporter = JaegerExporter(
            agent_host_name=config.jaeger_agent_host,
            agent_port=config.jaeger_agent_port,
        )
        logger.info(f"Using Jaeger exporter at {config.jaeger_agent_host}:{config.jaeger_agent_port}")
    elif config.tracing_exporter == "otlp" and OTLP_AVAILABLE:
        exporter = OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            insecure=True,  # Для development, в production использовать TLS
        )
        logger.info(f"Using OTLP exporter at {config.otlp_endpoint}")
    else:
        logger.warning(f"Tracing exporter '{config.tracing_exporter}' not available, tracing disabled")
        return
    
    # Добавить BatchSpanProcessor для эффективной отправки spans
    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)
    
    # Инструментировать FastAPI приложение
    FastAPIInstrumentor.instrument_app(app_instance)
    logger.info("OpenTelemetry tracing configured successfully")


# Создание FastAPI приложения
app = FastAPI(
    title="DataProcessor API",
    description="""
    API для управления обработкой видео через DataProcessor.
    
    ## Основные возможности
    
    * **Обработка видео**: Асинхронная обработка видео с поддержкой очереди
    * **Мониторинг статуса**: Получение детального статуса обработки и прогресса
    * **События в реальном времени**: Server-Sent Events (SSE) для отслеживания прогресса
    * **Артефакты**: Получение результатов обработки (NPZ файлы, manifest)
    * **Отмена обработки**: Мягкая отмена активных run'ов
    * **Health checks**: Проверка состояния API и зависимостей
    * **Метрики**: Prometheus метрики для мониторинга
    
    ## Аутентификация
    
    API использует API Key аутентификацию через заголовок `X-API-Key`.
    
    ## Версионирование
    
    API версионируется через префикс пути: `/api/v1/`
    
    ## Ошибки
    
    API возвращает стандартные HTTP коды статуса:
    * `200` - Успешный запрос
    * `202` - Запрос принят (асинхронная обработка)
    * `400` - Невалидный запрос
    * `401` - Требуется аутентификация
    * `403` - Невалидный API ключ
    * `404` - Ресурс не найден
    * `409` - Конфликт (например, дубликат run_id)
    * `410` - Ресурс больше не доступен (завершённый run)
    * `503` - Сервис недоступен (backpressure, unhealthy)
    
    ## Rate Limiting
    
    Endpoint `POST /api/v1/process` ограничен до 100 запросов в час на backend instance.
    Используйте заголовок `X-Backend-ID` для идентификации backend instance.
    
    ## Дополнительная информация
    
    * Swagger UI: `/docs`
    * ReDoc: `/redoc`
    * OpenAPI JSON: `/openapi.json`
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "DataProcessor API Support",
        "email": "support@dataprocessor.example.com",
    },
    license_info={
        "name": "Proprietary",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.dataprocessor.example.com",
            "description": "Production server"
        }
    ]
)

# Настройка rate limiting
if SLOWAPI_AVAILABLE and Limiter:
    try:
        limiter = Limiter(key_func=get_backend_id)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("Rate limiting enabled with backend ID support")
    except Exception as e:
        logger.warning(f"Failed to initialize rate limiting: {e}")
        limiter = None
else:
    limiter = None
    logger.warning("Rate limiting disabled (slowapi not available)")

# Добавление Request ID middleware (должен быть первым для добавления request_id во все запросы)
# Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2776-2785)
app.add_middleware(RequestIDMiddleware)
logger.info("Request ID middleware enabled")

# Добавление CORS middleware
# Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2149-2161)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS middleware configured with origins: {config.cors_origins_list}")

# Обработчики ошибок
@app.exception_handler(RunNotFoundError)
async def run_not_found_handler(request, exc: RunNotFoundError):
    """Обработчик для RunNotFoundError."""
    return JSONResponse(
        status_code=404,
        content={"error": "Run not found", "run_id": exc.run_id}
    )


@app.exception_handler(InvalidPayloadError)
async def invalid_payload_handler(request, exc: InvalidPayloadError):
    """Обработчик для InvalidPayloadError."""
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid payload", "details": exc.details}
    )


@app.exception_handler(ProcessingError)
async def processing_error_handler(request, exc: ProcessingError):
    """Обработчик для ProcessingError."""
    return JSONResponse(
        status_code=500,
        content={"error": "Processing failed", "message": str(exc)}
    )


@app.exception_handler(RunAlreadyExistsError)
async def run_already_exists_handler(request, exc: RunAlreadyExistsError):
    """Обработчик для RunAlreadyExistsError."""
    return JSONResponse(
        status_code=409,
        content={"error": "Run already exists", "run_id": exc.run_id}
    )


@app.exception_handler(BackpressureError)
async def backpressure_error_handler(request, exc: BackpressureError):
    """Обработчик для BackpressureError."""
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    return JSONResponse(
        status_code=503,
        content={"error": "Service overloaded", "message": str(exc)},
        headers=headers
    )


# Корневой endpoint
@app.get("/", tags=["root"])
async def root():
    """Корневой endpoint с информацией о сервисе."""
    return {
        "service": "DataProcessor API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Подключение роутеров
from api.endpoints import process, runs, health, artifacts, metrics, cancel, retention

app.include_router(process.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")
app.include_router(cancel.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
app.include_router(artifacts.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1")
app.include_router(retention.router, prefix="/api/v1")

# Инициализация OpenTelemetry tracing (опционально)
# Должна быть выполнена после создания app и роутеров
if OPENTELEMETRY_AVAILABLE and config.enable_tracing:
    try:
        _setup_opentelemetry(app)
        logger.info("OpenTelemetry tracing initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry tracing (continuing without tracing): {e}")
elif OPENTELEMETRY_AVAILABLE and not config.enable_tracing:
    logger.info("OpenTelemetry tracing disabled (enable_tracing=False)")
elif not OPENTELEMETRY_AVAILABLE:
    logger.debug("OpenTelemetry not available (packages not installed)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug,
        log_level=config.log_level.lower()
    )

