import onnxruntime as ort

model_path = "models/AASIST/aasist.onnx"

session = ort.InferenceSession(model_path)

print("MODEL INPUTS:")
for inp in session.get_inputs():
    print("Name:", inp.name)
    print("Shape:", inp.shape)
    print("Type:", inp.type)
    print()

print("MODEL OUTPUTS:")
for out in session.get_outputs():
    print("Name:", out.name)
    print("Shape:", out.shape)
    print("Type:", out.type)
    print()