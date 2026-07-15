-- Инициализация PostgreSQL для единого стека (docker-compose.prod.yml).
-- Создаёт БД/пользователя Fetcher рядом с основной БД trendflow.
-- Основная БД (trendflow) создаётся через POSTGRES_DB.

CREATE ROLE fetcher WITH LOGIN PASSWORD 'fetcher_password';
CREATE DATABASE fetcher_db OWNER fetcher;
GRANT ALL PRIVILEGES ON DATABASE fetcher_db TO fetcher;

-- БД Embedding Service (faiss-метаданные/объекты)
CREATE DATABASE embeddings OWNER trendflow;
