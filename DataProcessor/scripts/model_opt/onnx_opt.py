import onnx

def set_dynamic_batch(model):
    # Fix inputs shapes
    for t in model.graph.input:
        dim = t.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"

    # Fix outputs shapes
    for t in model.graph.output:
        dim = t.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"

    # ALSO fix all intermediate ValueInfo which Triton reads!
    for t in model.graph.value_info:
        dims = t.type.tensor_type.shape.dim
        if len(dims) > 0:
            dim = dims[0]
            if dim.dim_value == 1:
                dim.ClearField("dim_value")
                dim.dim_param = "batch"

    # Remove any leftover shape inference fixed dims
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                # leave tensors alone
                continue
            if attr.type == onnx.AttributeProto.INT:
                # not dimension
                continue
            if attr.type == onnx.AttributeProto.INTS:
                # not actual tensor shape
                continue

    return model


def patch_file(path_in, path_out=None):
    model = onnx.load(path_in)

    model = set_dynamic_batch(model)

    if path_out is None:
        path_out = path_in.replace(".onnx", "_dyn.onnx")

    onnx.save(model, path_out)
    print(f"[OK] Saved → {path_out}")


if __name__ == "__main__":
    import sys
    patch_file(sys.argv[1])
