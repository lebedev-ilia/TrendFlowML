#!/bin/bash
B=/workspace/triton-bundle
export LD_LIBRARY_PATH="$B/syslib:$B/cuda-12.6/targets/x86_64-linux/lib:$B/tritonserver/lib"
exec "$B/tritonserver/bin/tritonserver" \
  --model-repository=/workspace/TrendFlowML/DataProcessor/triton/models \
  --backend-directory="$B/tritonserver/backends" \
  --model-control-mode=explicit \
  --load-model=clip_image_224_onnx --load-model=clip_text_onnx --load-model=midas_256_onnx --load-model=raft_256_onnx --load-model=places365_resnet50_224_onnx \
  --load-model=preprocess_clip_image_224 --load-model=preprocess_midas_256 --load-model=preprocess_raft_256 --load-model=preprocess_places365_224 \
  --load-model=clip_image_224 --load-model=clip_text --load-model=midas_256 --load-model=raft_256 --load-model=places365_resnet50_224 \
  --http-port=8000 --allow-grpc=false --allow-metrics=false --log-verbose=0
