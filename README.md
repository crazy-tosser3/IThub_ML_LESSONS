mkdir -p third_party

curl -fL "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-osx-arm64-1.31.1.tgz" -o onnxruntime.tgz

tar -xzf onnxruntime.tgz
mv onnxruntime-osx-arm64-1.31.1 third_party/onnxruntime
rm onnxruntime.tgz
