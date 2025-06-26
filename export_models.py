import torch
import os
from model.model import Autoencoder1D
from config import CONFIG

os.makedirs("exported_models", exist_ok=True)

model = Autoencoder1D()
model.load_state_dict(torch.load(CONFIG["model_save_path"]))
model.eval()

dummy_input = torch.randn(1, 1, 1024)

# TorchScript (for Raspberry Pi / general CPU inference)
scripted_model = torch.jit.script(model)
scripted_model.save("exported_models/autoencoder_scripted.pt")
print("TorchScript model exported for Raspberry Pi at: exported_models/autoencoder_scripted.pt")

# ONNX (for TensorRT on Jetson)
torch.onnx.export(
    model,
    dummy_input,
    "exported_models/autoencoder.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)
print("ONNX model exported for TensorRT at: exported_models/autoencoder.onnx")
