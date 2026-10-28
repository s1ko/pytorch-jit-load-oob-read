import torch
import zipfile


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.arange(1 << 22, dtype=torch.float32))  # 4M floats = 16 MB

    def forward(self, x):
        return self.w.sum() + x


m = torch.jit.script(M())
torch.jit.save(m, "/tmp/clean2.pt")

z = zipfile.ZipFile("/tmp/clean2.pt")
print("entries:")
for n in z.namelist():
    print(f"  {n}  {z.getinfo(n).file_size}")
