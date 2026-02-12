# Start Comand

docker run --rm --gpus all --shm-size=1g   -p 8000:8000 -p 8001:8001 -p 8002:8002   -v "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/triton/{model_repo}:/models:ro"   nvcr.io/nvidia/tritonserver:24.08-py3   tritonserver --model-repository=/models
