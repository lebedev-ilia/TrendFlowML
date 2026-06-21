# Фаза 3: Контракт доступа к артефактам Fetcher

Документ фиксирует, как Backend и DataProcessor получают входное видео (и при необходимости мета/комментарии) из хранилища Fetcher (MinIO/S3) для run'ов ингестиции по YouTube URL.

См. также: [BACKEND_FETCHER_INTEGRATION_ANALYSIS.md](./BACKEND_FETCHER_INTEGRATION_ANALYSIS.md) (Фаза 3), [backend/docs/FETCHER_INTEGRATION.md](../backend/docs/FETCHER_INTEGRATION.md).

---

## 1. Источник артефактов

- **Fetcher** после успешного finalize кладёт артефакты в object storage (S3/MinIO) и отдаёт по запросу:
  - **GET /api/v1/runs/{run_id}/manifest** — структура manifest (video_id, platform, duration_seconds, пути к артефактам в storage).
  - **GET /api/v1/runs/{run_id}/artifacts** — список артефактов с **signed URL** (`download_url`) для скачивания (video_file, meta_file, comments_file и т.д.).

- **Backend** по `run_id` запрашивает у Fetcher manifest и artifacts, извлекает signed URL для `video_file` и либо скачивает видео во временный файл, либо передаёт URL в DataProcessor (если DataProcessor поддерживает вход по URL).

---

## 2. Варианты передачи видео в DataProcessor

| Вариант | Описание | Реализация |
|--------|----------|------------|
| **A** | Backend скачивает видео по signed URL во временный файл и передаёт в DataProcessor **локальный путь** (`video_path`). | Использовалось в Phase 2; остаётся fallback. |
| **B** | Backend передаёт в DataProcessor **video_url** (signed URL); DataProcessor сам скачивает видео в свой кэш (директория из `allowed_video_paths`) и далее работает с локальным путём. | Phase 3: DataProcessor API принимает опциональное поле `video_url`; при его наличии скачивание выполняет DataProcessor, Backend не держит файл. |
| **C** | Общий volume (MinIO смонтирован в Backend и DataProcessor); путь к файлу в storage передаётся как `video_path` (например `/mnt/minio/bucket/key`). | Требует инфраструктуры; при необходимости документировать отдельно. |

**Текущий выбор:** A + B. По умолчанию Backend может передавать `video_url` (вариант B); при отсутствии поддержки `video_url` на стороне DataProcessor или при ошибке Backend использует вариант A (скачивание во временный файл и передача `video_path`).

---

## 3. Контракт полей запроса к DataProcessor

- **video_path** (строка) — путь к видеофайлу на диске DataProcessor (обязателен, если не передан `video_url`). Должен находиться в разрешённых директориях (`allowed_video_paths`).
- **video_url** (строка, опционально) — URL для скачивания видео (например signed URL от Fetcher). Если передан, DataProcessor скачивает файл в кэш-директорию (внутри `allowed_video_paths`) и подставляет полученный локальный путь как `video_path` для пайплайна. Передавать либо `video_path`, либо `video_url` (приоритет у `video_url`, если заданы оба).

---

## 4. Кэш для видео по URL (DataProcessor)

- DataProcessor при получении `video_url` сохраняет скачанный файл в директорию кэша: **video_url_cache_dir** (конфиг, по умолчанию — поддиректория в первой из `allowed_video_paths`, например `{allowed_path}/_url_cache`).
- Имя файла в кэше: по возможности по `run_id` или по хэшу URL, чтобы избежать коллизий и при необходимости переиспользовать кэш.
- Очистка кэша (TTL, размер) может быть вынесена в отдельную политику; в MVP допускается ручная очистка.

---

## 5. Единая точка формирования payload в Backend

- Для **upload**-пути (AnalysisJob): по-прежнему **prepare_dataprocessor_payload(db, analysis_job)** — путь к видео берётся из `Video.storage_path` / legacy VideoFile.
- Для **ingestion**-пути (run из Fetcher): **build_ingestion_payload_from_fetcher(run_id, settings)** — запрос manifest и artifacts к Fetcher, извлечение signed URL для video_file; возврат структуры (platform_id, video_id, profile_config, **video_path** или **video_url**), совместимой с вызовом DataProcessor. Задача **process_ingestion_run** использует эту функцию и передаёт в run_dataprocessor_async либо локальный путь (после скачивания в Backend), либо video_url (если Backend не скачивает и передаёт URL в DataProcessor).

---

## 6. Связанные файлы

- Backend: `backend/app/services/fetcher_client.py` (get_run_manifest, get_run_artifacts), `backend/app/services/dataprocessor_adapter.py` (build_ingestion_payload_from_fetcher), `backend/app/services/dataprocessor.py` (run_dataprocessor_async с поддержкой video_url), `backend/app/tasks.py` (process_ingestion_run).
- DataProcessor: `DataProcessor/api/schemas/requests.py` (ProcessRequest.video_url), `DataProcessor/api/utils/validators.py`, `DataProcessor/api/endpoints/process.py` (скачивание по video_url в кэш), `DataProcessor/api/config.py` (video_url_cache_dir).
---

## Навигация

[Vault](MAIN_INDEX.md)
