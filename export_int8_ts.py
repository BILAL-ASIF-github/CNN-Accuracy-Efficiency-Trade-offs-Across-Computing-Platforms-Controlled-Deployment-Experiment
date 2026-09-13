import warnings, sys
warnings.filterwarnings("ignore")
import torch
from pathlib import Path

sys.path.insert(0, r"C:\Users\pc\Desktop\CNN Study")
import run_benchmark as R

INT8_EXPORT = Path(r"C:\Users\pc\Desktop\CNN Study\checkpoints\variants\variant_int8_ts.pt")

m = R.quantize_int8_fresh()
m.eval()

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

try:
    # Try TorchScript to get a self-contained, fresh-process-loadable artifact
    scripted = torch.jit.trace(Wrapper(m), torch.randn(1, 3, 224, 224))
    torch.jit.save(scripted, INT8_EXPORT)
    print(f"TorchScript export saved: {INT8_EXPORT} ({INT8_EXPORT.stat().st_size/1024/1024:.1f} MB)")
    export_mode = "torchscript"
except Exception as e:
    print(f"TorchScript trace failed: {e}")
    export_mode = None