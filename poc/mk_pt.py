import torch
import zipfile
import pickletools


class M(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        z = y * 2
        return z - 3


m = torch.jit.script(M())
torch.jit.save(m, "/tmp/clean.pt")

z = zipfile.ZipFile("/tmp/clean.pt")
print("ZIP entries:")
for n in z.namelist():
    print("  ", n, z.getinfo(n).file_size)

dbg = [n for n in z.namelist() if n.endswith(".debug_pkl")]
print("\ndebug_pkl files:", dbg)
for d in dbg:
    raw = z.read(d)
    print(f"\n--- {d} ({len(raw)} bytes) pickletools.dis ---")
    pickletools.dis(raw)
