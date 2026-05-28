# Примеры использования DataProcessor API

Практические примеры использования DataProcessor API на различных языках программирования.

## Содержание

1. [Python](#python)
2. [JavaScript/TypeScript](#javascripttypescript)
3. [cURL](#curl)
4. [Полный пример обработки](#полный-пример-обработки)

## Python

### Базовый пример

```python
import httpx
import asyncio
import time

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"

async def process_video():
    async with httpx.AsyncClient() as client:
        # 1. Запуск обработки
        response = await client.post(
            f"{API_URL}/api/v1/process",
            headers={"X-API-Key": API_KEY},
            json={
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "video_id": "video-123",
                "platform_id": "youtube",
                "video_path": "/path/to/video.mp4",
                "config_hash": "abc123",
                "profile_config": {
                    "visual": {"enabled": True},
                    "audio": {"enabled": True},
                    "text": {"enabled": True}
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        run_id = data["run_id"]
        print(f"Processing started: {run_id}")
        
        # 2. Отслеживание статуса
        while True:
            status_response = await client.get(
                f"{API_URL}/api/v1/runs/{run_id}/status",
                headers={"X-API-Key": API_KEY}
            )
            status_response.raise_for_status()
            status = status_response.json()
            
            print(f"Status: {status['status']}, Progress: {status.get('progress', 0):.2%}")
            
            if status["status"] in ["success", "error", "cancelled"]:
                break
            
            await asyncio.sleep(5)
        
        # 3. Получение результатов
        if status["status"] == "success":
            manifest_response = await client.get(
                f"{API_URL}/api/v1/runs/{run_id}/manifest",
                headers={"X-API-Key": API_KEY}
            )
            manifest_response.raise_for_status()
            manifest = manifest_response.json()
            print(f"Manifest: {manifest}")

asyncio.run(process_video())
```

### Отслеживание через SSE

```python
import httpx
import asyncio
import json

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"

async def track_with_sse(run_id: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"{API_URL}/api/v1/runs/{run_id}/events",
            headers={"X-API-Key": API_KEY}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    event_type = data.get("event_type", "status")
                    
                    if event_type == "status":
                        print(f"Status: {data['status']}, Progress: {data.get('progress', 0):.2%}")
                    elif event_type == "stage":
                        print(f"Stage: {data['stage']} - {data['status']}")
                    elif event_type == "component":
                        print(f"Component: {data['component']} - {data['status']}")
                    elif event_type == "log":
                        print(f"Log [{data['level']}]: {data['message']}")
                    
                    if data.get("status") in ["success", "error", "cancelled"]:
                        break

# Использование
asyncio.run(track_with_sse("550e8400-e29b-41d4-a716-446655440000"))
```

### Получение артефактов

```python
import httpx
import asyncio

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"

async def download_artifacts(run_id: str):
    async with httpx.AsyncClient() as client:
        artifacts = ["visual.npz", "audio.npz", "text.npz"]
        
        for artifact in artifacts:
            response = await client.get(
                f"{API_URL}/api/v1/runs/{run_id}/artifacts/{artifact}",
                headers={"X-API-Key": API_KEY}
            )
            response.raise_for_status()
            
            with open(artifact, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {artifact}")

asyncio.run(download_artifacts("550e8400-e29b-41d4-a716-446655440000"))
```

## JavaScript/TypeScript

### Базовый пример

```typescript
const API_URL = 'http://localhost:8000';
const API_KEY = 'your-api-key';

async function processVideo() {
  // 1. Запуск обработки
  const processResponse = await fetch(`${API_URL}/api/v1/process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY
    },
    body: JSON.stringify({
      run_id: '550e8400-e29b-41d4-a716-446655440000',
      video_id: 'video-123',
      platform_id: 'youtube',
      video_path: '/path/to/video.mp4',
      config_hash: 'abc123',
      profile_config: {
        visual: { enabled: true },
        audio: { enabled: true },
        text: { enabled: true }
      }
    })
  });

  if (!processResponse.ok) {
    throw new Error(`Failed to start processing: ${processResponse.statusText}`);
  }

  const { run_id } = await processResponse.json();
  console.log(`Processing started: ${run_id}`);

  // 2. Отслеживание статуса
  while (true) {
    const statusResponse = await fetch(
      `${API_URL}/api/v1/runs/${run_id}/status`,
      {
        headers: { 'X-API-Key': API_KEY }
      }
    );

    if (!statusResponse.ok) {
      throw new Error(`Failed to get status: ${statusResponse.statusText}`);
    }

    const status = await statusResponse.json();
    console.log(`Status: ${status.status}, Progress: ${(status.progress || 0) * 100}%`);

    if (['success', 'error', 'cancelled'].includes(status.status)) {
      break;
    }

    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  // 3. Получение результатов
  const manifestResponse = await fetch(
    `${API_URL}/api/v1/runs/${run_id}/manifest`,
    {
      headers: { 'X-API-Key': API_KEY }
    }
  );

  if (!manifestResponse.ok) {
    throw new Error(`Failed to get manifest: ${manifestResponse.statusText}`);
  }

  const manifest = await manifestResponse.json();
  console.log('Manifest:', manifest);
}

processVideo().catch(console.error);
```

### Отслеживание через SSE

```typescript
const API_URL = 'http://localhost:8000';
const API_KEY = 'your-api-key';

function trackWithSSE(runId: string) {
  const eventSource = new EventSource(
    `${API_URL}/api/v1/runs/${runId}/events?api_key=${API_KEY}`
  );

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const eventType = data.event_type || 'status';

    switch (eventType) {
      case 'status':
        console.log(`Status: ${data.status}, Progress: ${(data.progress || 0) * 100}%`);
        break;
      case 'stage':
        console.log(`Stage: ${data.stage} - ${data.status}`);
        break;
      case 'component':
        console.log(`Component: ${data.component} - ${data.status}`);
        break;
      case 'log':
        console.log(`Log [${data.level}]: ${data.message}`);
        break;
    }

    if (['success', 'error', 'cancelled'].includes(data.status)) {
      eventSource.close();
    }
  };

  eventSource.onerror = (error) => {
    console.error('SSE error:', error);
    eventSource.close();
  };
}

trackWithSSE('550e8400-e29b-41d4-a716-446655440000');
```

### React Hook для отслеживания

```typescript
import { useState, useEffect } from 'react';

function useRunStatus(runId: string | null) {
  const [status, setStatus] = useState<any>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!runId) return;

    const eventSource = new EventSource(
      `${API_URL}/api/v1/runs/${runId}/events?api_key=${API_KEY}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data);
    };

    eventSource.onerror = (err) => {
      setError(new Error('SSE connection failed'));
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [runId]);

  return { status, error };
}

// Использование
function VideoProcessingComponent({ runId }: { runId: string }) {
  const { status, error } = useRunStatus(runId);

  if (error) return <div>Error: {error.message}</div>;
  if (!status) return <div>Loading...</div>;

  return (
    <div>
      <p>Status: {status.status}</p>
      <p>Progress: {(status.progress || 0) * 100}%</p>
      {status.current_stage && <p>Stage: {status.current_stage}</p>}
    </div>
  );
}
```

## cURL

### Запуск обработки

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "video_id": "video-123",
    "platform_id": "youtube",
    "video_path": "/path/to/video.mp4",
    "config_hash": "abc123",
    "profile_config": {
      "visual": {"enabled": true},
      "audio": {"enabled": true},
      "text": {"enabled": true}
    }
  }'
```

### Получение статуса

```bash
curl "http://localhost:8000/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/status" \
  -H "X-API-Key: your-api-key"
```

### Отслеживание через SSE

```bash
curl -N "http://localhost:8000/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/events?api_key=your-api-key"
```

### Получение manifest

```bash
curl "http://localhost:8000/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/manifest" \
  -H "X-API-Key: your-api-key"
```

### Получение артефактов

```bash
curl "http://localhost:8000/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/artifacts/visual.npz" \
  -H "X-API-Key: your-api-key" \
  -o visual.npz
```

### Отмена обработки

```bash
curl -X POST "http://localhost:8000/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/cancel" \
  -H "X-API-Key: your-api-key"
```

## Полный пример обработки

### Python класс для работы с API

```python
import httpx
import asyncio
import json
from typing import Optional, Dict, Any

class DataProcessorClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    async def process_video(
        self,
        run_id: str,
        video_id: str,
        platform_id: str,
        video_path: str,
        config_hash: str,
        profile_config: Dict[str, Any]
    ) -> str:
        """Запустить обработку видео."""
        response = await self.client.post(
            f"{self.api_url}/api/v1/process",
            headers={"X-API-Key": self.api_key},
            json={
                "run_id": run_id,
                "video_id": video_id,
                "platform_id": platform_id,
                "video_path": video_path,
                "config_hash": config_hash,
                "profile_config": profile_config
            }
        )
        response.raise_for_status()
        return response.json()["run_id"]
    
    async def get_status(self, run_id: str) -> Dict[str, Any]:
        """Получить статус обработки."""
        response = await self.client.get(
            f"{self.api_url}/api/v1/runs/{run_id}/status",
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()
    
    async def wait_for_completion(
        self,
        run_id: str,
        poll_interval: int = 5,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """Ожидать завершения обработки."""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            status = await self.get_status(run_id)
            
            if status["status"] in ["success", "error", "cancelled"]:
                return status
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Processing timeout after {timeout} seconds")
            
            await asyncio.sleep(poll_interval)
    
    async def get_manifest(self, run_id: str) -> Dict[str, Any]:
        """Получить manifest."""
        response = await self.client.get(
            f"{self.api_url}/api/v1/runs/{run_id}/manifest",
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()
    
    async def download_artifact(
        self,
        run_id: str,
        artifact_path: str,
        output_path: str
    ):
        """Скачать артефакт."""
        response = await self.client.get(
            f"{self.api_url}/api/v1/runs/{run_id}/artifacts/{artifact_path}",
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
    
    async def cancel(self, run_id: str) -> Dict[str, Any]:
        """Отменить обработку."""
        response = await self.client.post(
            f"{self.api_url}/api/v1/runs/{run_id}/cancel",
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Закрыть клиент."""
        await self.client.aclose()

# Использование
async def main():
    client = DataProcessorClient("http://localhost:8000", "your-api-key")
    
    try:
        # Запуск обработки
        run_id = await client.process_video(
            run_id="550e8400-e29b-41d4-a716-446655440000",
            video_id="video-123",
            platform_id="youtube",
            video_path="/path/to/video.mp4",
            config_hash="abc123",
            profile_config={
                "visual": {"enabled": True},
                "audio": {"enabled": True},
                "text": {"enabled": True}
            }
        )
        
        # Ожидание завершения
        final_status = await client.wait_for_completion(run_id)
        
        if final_status["status"] == "success":
            # Получение manifest
            manifest = await client.get_manifest(run_id)
            print(f"Manifest: {manifest}")
            
            # Скачивание артефактов
            await client.download_artifact(run_id, "visual.npz", "visual.npz")
            await client.download_artifact(run_id, "audio.npz", "audio.npz")
            await client.download_artifact(run_id, "text.npz", "text.npz")
        else:
            print(f"Processing failed: {final_status}")
    finally:
        await client.close()

asyncio.run(main())
```

## Обработка ошибок

### Python

```python
import httpx

async def process_with_error_handling():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/process",
                headers={"X-API-Key": "your-api-key"},
                json={...}
            )
            
            if response.status_code == 409:
                print("Run already exists")
            elif response.status_code == 503:
                print("Service overloaded, retry later")
            elif response.status_code == 429:
                print("Rate limit exceeded")
            else:
                response.raise_for_status()
                
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Request error: {e}")
```

## Примечания

- Все примеры используют асинхронные запросы для лучшей производительности
- Для production рекомендуется использовать connection pooling и retry логику
- API Key должен храниться в безопасном месте (environment variables, secrets manager)
- Для длительных операций рекомендуется использовать SSE для real-time обновлений

