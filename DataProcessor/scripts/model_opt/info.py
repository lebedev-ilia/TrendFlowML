# save как inspect_onnx.py и запустите: python inspect_onnx.py /path/to/yolo11x_640.onnx
import sys
import onnx

def shape(d):
    out = []
    for dim in d.type.tensor_type.shape.dim:
        # динамический или неизвестный размер часто представлен как dim_value=0 или dim_param set
        v = getattr(dim, "dim_value", None)
        if v is None or v == 0:
            out.append(None)
        else:
            out.append(int(v))
    return out

m = onnx.load("/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/triton/models_yolo11x_640/yolo11x_640/1/model.onnx")
print("Inputs:")
for i in m.graph.input:
    print(" ", i.name, "shape:", shape(i))
print("Outputs:")
for o in m.graph.output:
    print(" ", o.name, "shape:", shape(o))
print("Opset imports:", [(oi.domain, oi.version) for oi in m.opset_import])