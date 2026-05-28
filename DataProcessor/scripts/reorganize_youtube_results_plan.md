# План реорганизации dp_results/youtube

## Текущая проблема

В `dp_results/youtube/` находится 42+ директорий с результатами в плоской структуре:
- `test_action_recognition_*` (20 директорий)
- `audit3_*_smoke_*` (11 директорий)
- `test_*_audit_3` (5 директорий)
- `video*` (старые результаты)

Это затрудняет навигацию и управление результатами.

## Предлагаемая структура

```
youtube/
├── tests/                          # Тестовые прогоны
│   └── action_recognition/         # Тесты action_recognition
│       ├── test_action_recognition_shortest/
│       ├── test_action_recognition_v2/
│       └── ...
│
├── audit/                          # Результаты аудита
│   └── v3/                         # Audit v3
│       ├── components/             # Тесты компонентов
│       │   ├── test_1_audit_3/
│       │   ├── test_2_audit_3/
│       │   └── ...
│       └── smoke/                  # Smoke тесты
│           ├── audit3_action_recognition_smoke_2/
│           ├── audit3_cod_smoke_1/
│           └── ...
│
├── archive/                         # Архив старых результатов
│   ├── old_videos/                 # Старые видео
│   │   ├── video1/
│   │   ├── video2/
│   │   └── ...
│   └── old_tests/                  # Старые тесты (если есть)
│
└── README.md                        # Документация структуры
```

## Преимущества

1. **Логическая группировка**: результаты сгруппированы по назначению
2. **Легкая навигация**: проще найти нужные результаты
3. **Масштабируемость**: легко добавлять новые категории
4. **Чистота**: старые результаты в архиве

## Маппинг директорий

### tests/action_recognition/
- `test_action_recognition_shortest` → `tests/action_recognition/test_action_recognition_shortest`
- `test_action_recognition_v2` → `tests/action_recognition/test_action_recognition_v2`
- ... (все 20 директорий)

### audit/v3/smoke/
- `audit3_action_recognition_smoke_2` → `audit/v3/smoke/audit3_action_recognition_smoke_2`
- `audit3_cod_smoke_1` → `audit/v3/smoke/audit3_cod_smoke_1`
- ... (все smoke тесты)

### audit/v3/components/
- `test_1_audit_3` → `audit/v3/components/test_1_audit_3`
- `test_2_audit_3` → `audit/v3/components/test_2_audit_3`
- ... (все тесты аудита)

### archive/old_videos/
- `video1` → `archive/old_videos/video1`
- `video2` → `archive/old_videos/video2`
- `video3` → `archive/old_videos/video3`
- `test_video_1` → `archive/old_videos/test_video_1`

## Безопасность

- ✅ Создаётся резервная копия перед реорганизацией
- ✅ Проверка существования директорий перед перемещением
- ✅ Сохранение всех данных

## Выполнение

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
bash scripts/reorganize_youtube_results.sh
```

## После реорганизации

1. Обновить пути в скриптах (если используются абсолютные пути)
2. Обновить документацию с новыми путями
3. Проверить, что все ссылки работают

## Откат

Если что-то пошло не так, можно восстановить из резервной копии:
```bash
rm -rf /media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/youtube/*
cp -r /media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/youtube_backup_*/youtube/* \
      /media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/youtube/
```

