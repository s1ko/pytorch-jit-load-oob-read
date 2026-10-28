# PoC files

Two crafted TorchScript archives, each derived from a trivial scripted module. Both are loaded with `torch.jit.load`.

| File | Purpose |
|---|---|
| `mk_pt.py` | Scripts a small module and writes `/tmp/clean.pt`. |
| `craft_pt.py` | Rewrites one `textIndex` entry in `clean.pt`'s `.debug_pkl` from `4` to `0x7fffffff` and writes `/tmp/evil_jit.pt`. |
| `evil_jit.pt` | Output of the two scripts above (1617 bytes). Instance 1. |
| `mk_pt2.py` | Scripts a module with a 16 MB buffer and writes `/tmp/clean2.pt`. |
| `craft_storage.py` | Truncates the `data/0` record of `clean2.pt` to 16 bytes and writes `evil_storage.pt` in the current directory. Run it from `/tmp`. |
| `evil_storage.pt` | Output of the two scripts above (1771 bytes). Instance 2. |
| `scan_torchscript_archive.py` | Standard-library scanner for both conditions. Does not import `torch` and does not execute pickled code. |
| `repro-textindex.txt`, `repro-storage.txt` | Console output recorded on `torch 2.6.0+cu124` (exit status 139 for the first). |

Regenerate:

```bash
cd /tmp
python mk_pt.py && python craft_pt.py
python mk_pt2.py && python craft_storage.py
```

Trigger, in a fresh process each time (needs `torch`):

```python
import torch
m = torch.jit.load("evil_jit.pt")
print(m.code)

m = torch.jit.load("evil_storage.pt")
print(m(torch.ones(1)))
```

Scan:

```bash
python3 -I scan_torchscript_archive.py evil_jit.pt
python3 -I scan_torchscript_archive.py evil_storage.pt
```

Exit status 1 and a `SUSPICIOUS` line per finding.

Both archives embed `/tmp/mk_pt.py` and `/tmp/mk_pt2.py` as the original script paths in their debug info. Do not load them in a process you care about.
