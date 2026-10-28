import io
import pickle
import sys
import zipfile

ITEMSIZE = {
    "DoubleStorage": 8,
    "FloatStorage": 4,
    "HalfStorage": 2,
    "BFloat16Storage": 2,
    "LongStorage": 8,
    "IntStorage": 4,
    "ShortStorage": 2,
    "CharStorage": 1,
    "ByteStorage": 1,
    "BoolStorage": 1,
    "ComplexDoubleStorage": 16,
    "ComplexFloatStorage": 8,
}


class Loader(pickle.Unpickler):
    def __init__(self, data):
        super().__init__(io.BytesIO(data))
        self.storages = []

    def find_class(self, module, name):
        if name == "build_intlist":
            return list
        return type(name, (), {"__init__": lambda self, *args, **kwargs: None, "__setstate__": lambda self, state: None})

    def persistent_load(self, pid):
        self.storages.append(pid)
        return None


def walk(obj, table_len, hits, path, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, tuple):
        if len(obj) == 3 and isinstance(obj[0], list) and isinstance(obj[1], int) and isinstance(obj[2], int):
            for index in obj[0]:
                if isinstance(index, int) and not 0 <= index < table_len:
                    hits.add((path, index, table_len))
        for item in obj:
            walk(item, table_len, hits, path, seen)
    elif isinstance(obj, list):
        for item in obj:
            walk(item, table_len, hits, path, seen)


def check_debug_info(archive, name, findings):
    root = Loader(archive.read(name)).load()
    if not (isinstance(root, tuple) and len(root) == 3 and root[0] == "FORMAT_WITH_STRING_TABLE"):
        return
    table_len = len(root[1])
    hits = set()
    walk(root[2], table_len, hits, name, set())
    for path, index, size in sorted(hits):
        findings.append("%s: textIndex %d outside string table of %d entries" % (path, index, size))


def check_storages(archive, name, findings):
    prefix = name.rsplit("/", 1)[0]
    loader = Loader(archive.read(name))
    loader.load()
    sizes = {i.filename: i.file_size for i in archive.infolist()}
    for pid in loader.storages:
        if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
            continue
        kind = getattr(pid[1], "__name__", "")
        key, numel = pid[2], pid[4]
        record = sizes.get("/".join((prefix, "data", key)))
        itemsize = ITEMSIZE.get(kind.replace("Untyped", "Byte"))
        if record is None or itemsize is None or not isinstance(numel, int):
            continue
        if numel * itemsize > record:
            findings.append(
                "%s: storage %s declares %d bytes but record holds %d" % (name, key, numel * itemsize, record)
            )


def main(path):
    findings = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".debug_pkl"):
                check_debug_info(archive, name, findings)
            elif name.endswith("/data.pkl"):
                check_storages(archive, name, findings)
    for line in findings:
        print("SUSPICIOUS", line)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
