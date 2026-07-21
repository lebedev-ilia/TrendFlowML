#!/bin/bash
set -e
BUNDLE=/workspace/triton-bundle
IMAGE=nvcr.io/nvidia/tritonserver:24.08-py3
mkdir -p "$BUNDLE/syslib"
if [ -x "$BUNDLE/tritonserver/bin/tritonserver" ]; then echo "bundle exists"; exit 0; fi
echo "== docker pull $IMAGE =="; docker pull "$IMAGE"
echo "== extract tritonserver + cuda =="
docker rm -f tb_extract 2>/dev/null || true
docker create --name tb_extract "$IMAGE"
docker cp tb_extract:/opt/tritonserver "$BUNDLE/tritonserver"
docker cp tb_extract:/usr/local/cuda-12.6 "$BUNDLE/cuda-12.6"
docker rm tb_extract
echo "== runtime libs =="
docker rm -f tb_libs 2>/dev/null || true
docker create --name tb_libs "$IMAGE"
for lib in libcudnn.so.9.3.0 libcudnn.so.9 libcudnn.so libcublas.so.12 libcublasLt.so.12 \
  libpython3.10.so.1.0 libpython3.10.so.1 libpython3.10.so libicuuc.so.70.1 libicudata.so.70.1 \
  libicui18n.so.70.1 libb64.so.0d libdcgm.so.3.2.6 libxml2.so.2.9.13; do
  docker cp "tb_libs:/usr/lib/x86_64-linux-gnu/$lib" "$BUNDLE/syslib/" 2>/dev/null || true
done
docker rm tb_libs
cd "$BUNDLE/syslib"
ln -sf libcudnn.so.9.3.0 libcudnn.so.9; ln -sf libpython3.10.so.1.0 libpython3.10.so.1
ln -sf libicuuc.so.70.1 libicuuc.so.70; ln -sf libicudata.so.70.1 libicudata.so.70
ln -sf libxml2.so.2.9.13 libxml2.so.2; ln -sf libdcgm.so.3.2.6 libdcgm.so.3
echo "SETUP_DONE $BUNDLE"
