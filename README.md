# PyTorch: out-of-bounds reads when loading a TorchScript archive

`torch.jit.load` trusts two kinds of values read from the archive that index or size memory: the per-element indices of the source-range string table in `*.py.debug_pkl`, and the element count declared for a tensor storage in `data.pkl`. Neither is checked against the real size of what it indexes. A crafted `.pt` therefore makes the loader read past a `std::vector` of `shared_ptr<std::string>` (and dereference the garbage pointer), or build a `Storage` that claims more bytes than its backing zip record holds. In both cases the process crashes with SIGSEGV.

> ## ⚠ Unpatched as of 2026-10-28
> Both sites are unchanged in `v2.15.0` (the latest release) and in `main`. Upstream policy treats TorchScript archives as executable code and lists crashes and out-of-bounds access as non-security bugs; TorchScript is deprecated in favour of `torch.export`. Do not `torch.jit.load` files you do not trust. Mitigation below.

| | |
|---|---|
| CVE | None assigned |
| CWE | CWE-125 Out-of-bounds Read (instance 1 also CWE-129, instance 2 also CWE-130) |
| Project | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| Affected | Crash reproduced on `torch 2.6.0+cu124` by the reporter; reported by a third party on 2.7.0 and 2.11.0 (instance 1); source identical in `v2.15.0` (`31a77fc9c0a7`) and `main` (`94fc3eee`). |
| Fixed in | Not fixed |
| CVSS v3.1 | `CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:N/A:H` = 5.5 (Medium), crash only |
| Reported by | s1ko ([github.com/s1ko](https://github.com/s1ko), `s1ko@riseup.net`) |

## Scope, stated plainly

- Upstream position. `SECURITY.md` at `v2.15.0` lists, under "Issues That Are Not Security Vulnerabilities": "**Crashes and out-of-bounds access** ... bugs worth reporting as regular issues, but not security vulnerabilities", and "**Malformed de-serialized objects** ... such objects can lead to crash and out-of-bound access which are regular issues, not security vulnerabilities". Its "TorchScript models" section says "TorchScript models should be treated the same way as locally executable code from an unknown source. Only run TorchScript models if you trust the provider." On the related issues a maintainer wrote that TorchScript "has been deprecated in favor of torch.export as of 2.11".
- Consequence. Someone who can hand you a TorchScript model can already make you run the model's code. The additional impact of these bugs is limited to what a malformed archive does to the loader itself: a reliable crash of the loading process, and, for instance 2, out-of-bounds heap reads whose values flow into tensors. Disclosure of those values and any write primitive are not demonstrated. UNVERIFIED.
- Instance 1 is also public. See [Prior coverage](#prior-coverage): the same defect was published three days before the report to Huntr. This repository is an independent report of it, not a first disclosure. Instance 2 had no public coverage found.
- Demonstrated: SIGSEGV (exit 139) on `torch 2.6.0+cu124` for both archives, recorded in `poc/repro-*.txt`. Not re-run on 2.15.0 for this write-up; the claim that 2.15.0 and `main` are affected rests on reading the unchanged source. UNVERIFIED at runtime.

## Threat model

Attacker supplies a `.pt` (TorchScript zip). The victim loads it with `torch.jit.load` / `torch::jit::load` and then either inspects it, re-exports it, or runs it. Typical exposure: model-conversion or optimisation services, model hubs that load uploads for inspection, notebooks.

## Execution

All excerpts are from `v2.15.0` (`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`). `main` at `94fc3eeee663` has the same code at the same lines.

### Instance 1: source-range string table (`textIndex`)

`torch/csrc/jit/serialization/source_range_serialization.cpp:70-85`, `SourceRangeDeserializer::deserialize_source`:

```cpp
    int64_t fnameIndex = tup_elems[1].toInt();
    int64_t starting_line_no_ = tup_elems[2].toInt();
    std::optional<std::string> filename = std::nullopt;

    TORCH_CHECK(
        (uint64_t)fnameIndex < text_table_.size(),
        "Text table index is out of range")
    filename = *text_table_[fnameIndex];

    std::vector<std::string_view> pieces;
    std::vector<std::shared_ptr<std::string>> strs;

    for (int64_t i : textIndex) {
      pieces.emplace_back(*text_table_[i]);
      strs.emplace_back(text_table_[i]);
    }
```

`fnameIndex` is bounds-checked (lines 74-76); the elements of `textIndex` (lines 82-84) are not. Both come from the unpickled `.debug_pkl`. `text_table_` is a `std::vector<std::shared_ptr<std::string>>`, so an out-of-range `i` reads a `shared_ptr` from beyond the vector, and `*text_table_[i]` dereferences it.

Source ranges are unpickled lazily. `torch.jit.load` alone does not crash. The read happens the first time a source range is materialised: `module.code`, `module.inlined_graph`, `torch.jit.save(module, ...)`. Calling the module (`m(x)`) does not materialise them and did not crash.

### Instance 2: storage size versus record size

`torch/csrc/jit/serialization/import_read.cpp:40-43`:

```cpp
  auto read_record = [&](const std::string& name) {
    std::string ss = tensor_dir_path + name;
    return std::get<0>(stream_reader.getRecord(ss));
  };
```

`PyTorchStreamReader::getRecord` returns `(DataPtr, size_t)`, the second member being the real uncompressed size of the zip record. The lambda discards it.

`torch/csrc/jit/serialization/unpickler.cpp:572-596`, storage branch of `persistent_load`:

```cpp
        int64_t numel = args.at(4).toInt();
        size_t nbytes = 0;
        auto dtype = scalarTypeToTypeMeta(type);

        TORCH_CHECK(numel >= 0, "Numel can not be negative");
        TORCH_CHECK(
            !c10::mul_overflows(
                static_cast<size_t>(numel), dtype.itemsize(), &nbytes),
            "Tensor storage size overflowed");

        at::DataPtr storage_ptr;
        if (numel > 0) {
          // If there are no elements in the tensor, there's no point in
          // reading a zero (0) byte file from the input stream and paying
          // that cost.
          storage_ptr = read_record_(key);
        }

        storage = at::Storage(
            c10::Storage::use_byte_size_t(),
            nbytes,
            std::move(storage_ptr),
```

`numel` comes from `data.pkl`; `nbytes` is checked only for sign and multiplication overflow. The buffer comes from the zip record. Nothing requires `nbytes <= record size`. The later checks on tensor views (`unpickler.cpp:1016-1041`) compare against `storage_tensor.storage().nbytes()`, which is the claimed `nbytes`, so they do not catch it either. A `Storage` claiming 16 MB over a 16-byte buffer is the result, and any read of the tensor runs past the buffer. Unlike instance 1, this is reached by running the model (`m(x)`), by `m.buffers()` and by reading a parameter.

### Reproduction

Archives are in `poc/`; scripts to regenerate them are alongside. See [`poc/README.md`](poc/README.md).

Instance 1 (`poc/craft_pt.py` changes one `textIndex` element from `BININT1 4` to `BININT 0x7fffffff`):

```python
import torch
m = torch.jit.load("evil_jit.pt")
print(m.code)
```

Observed on `torch 2.6.0+cu124` (per trigger, fresh process each):

```
torch.jit.load() alone   -> exit 0
m.code                   -> exit 139 (SIGSEGV)
m.inlined_graph          -> exit 139 (SIGSEGV)
torch.jit.save(m, ...)   -> exit 139 (SIGSEGV)
m(torch.ones(3))         -> exit 0
```

Instance 2 (`poc/mk_pt2.py` creates a module with a 4M-element float buffer; `poc/craft_storage.py` cuts the 16 MB `data/0` record to 16 bytes while `data.pkl` still declares `numel = 4194304`):

```python
import torch
m = torch.jit.load("evil_storage.pt")
print(m(torch.ones(1)))
```

Observed on `torch 2.6.0+cu124`:

```
torch.jit.load() alone        -> exit 0
m(torch.ones(1))              -> exit 139 (SIGSEGV)
list(m.buffers()); b.sum()    -> exit 139 (SIGSEGV)
m.w.sum()                     -> exit 139 (SIGSEGV)
```

### Suggested fix

Instance 1: bounds-check every element, as is already done for `fnameIndex`.

```cpp
    for (int64_t i : textIndex) {
      TORCH_CHECK(
          i >= 0 && (uint64_t)i < text_table_.size(),
          "Text table index is out of range");
      pieces.emplace_back(*text_table_[i]);
      strs.emplace_back(text_table_[i]);
    }
```

Instance 2: keep the record size returned by `getRecord` and require `nbytes <= record_size` before constructing the `Storage`.

## Detection

File-format checks, no `torch` needed, no pickled code executed (`poc/scan_torchscript_archive.py`):

1. For every `*.debug_pkl` whose root is `("FORMAT_WITH_STRING_TABLE", string_table, ranges)`: every integer in a source tuple's `textIndex` list must be in `[0, len(string_table))`.
2. For every `*/data.pkl`: for each `persistent_load` id `("storage", <type>, key, location, numel)`, `numel * itemsize` must not exceed the size of the zip member named `data/<key>` inside the archive's top-level directory.

```
$ python3 -I poc/scan_torchscript_archive.py poc/evil_jit.pt
SUSPICIOUS clean/code/__torch__.py.debug_pkl: textIndex 2147483647 outside string table of 6 entries
$ python3 -I poc/scan_torchscript_archive.py poc/evil_storage.pt
SUSPICIOUS clean2/data.pkl: storage 0 declares 16777216 bytes but record holds 16
```

Both checks were also run against copies of the PoCs with the corruption reverted (`4` restored; record padded to 16 MB) and returned exit status 0. The scanner understands the TorchScript layouts produced by `torch 2.6.0`; other layouts are not covered and a clean result is not proof of safety.

Runtime signatures:

- SIGSEGV in a process that called `torch.jit.load` and then touched `.code`, `.inlined_graph`, `torch.jit.save`, a buffer or a parameter. Top frame for instance 1: `torch::jit::SourceRangeDeserializer::deserialize_source`.
- Under AddressSanitizer in a source build, instance 1 reports a read of a `shared_ptr` slot past the end of the `text_table_` allocation, instance 2 a heap read past the allocation returned by `getRecord`. These ASan traces were not captured for this write-up. UNVERIFIED.
- Core dumps of a Python process with frames in `torch::jit::Unpickler::run` or `SourceRangeDeserializer` shortly after a model upload are worth triaging.

## Mitigation

- No upstream fix exists. Do not call `torch.jit.load` / `torch::jit::load` on files from untrusted sources, and do not run inspection or re-export tooling (`.code`, `.inlined_graph`, `torch.jit.save`, `torch.utils.model_dump`) on them either.
- Prefer formats that carry no code or in-file sizes you must trust: `safetensors` for weights; `torch.load(..., weights_only=True)` for state dicts; `torch.export` for exported programs.
- If untrusted TorchScript must be handled, do it in a disposable, memory-limited, network-less process, and run `poc/scan_torchscript_archive.py` first.
- Downstream builds can carry the two checks under "Suggested fix".
- Mapping: CWE-125 / CWE-129 (validate indices and sizes read from files before use); ATT&CK M1048 Application Isolation and Sandboxing; D3FEND Process Isolation.

## Timeline

| Date (UTC) | Event |
|---|---|
| 2026-04-07 | pytorch/pytorch issue #179581 (professor-moody) lists four crash sites in `torch::jit::load`, one in `SourceRangeDeserializer::deserialize_source()`, without a root cause. A maintainer replies that TorchScript is deprecated; closed the same day. |
| 2026-05-27 | Issue #185441 (professor-moody) publishes instance 1 with root cause and fix. |
| 2026-05-30 | Both instances reported via huntr.com (report `cc995b82-617c-460b-987c-da2f3e389b91`). No triage, validation or response from the platform as of 2026-10-28. |
| 2026-06-12 | PR #187153 (Arindamsamanta2004) adds the `textIndex` check. |
| 2026-08-07 | #185441 closed as "not planned": "TorchScript has been deprecated in favor of torch.export as of 2.11". |
| 2026-09-21 | PR #187153 closed unmerged by the stale bot. |
| 2026-10-28 | `v2.15.0` released; source of both sites unchanged in it and in `main`. This advisory published. |

## Prior coverage

- Instance 1 was published before this report. [#185441](https://github.com/pytorch/pytorch/issues/185441), opened 2026-05-27 by professor-moody, describes the same unchecked `textIndex` loop in `deserialize_source`, confirms it on 2.7.0 and 2.11.0, and proposes the same fix. That is three days before the Huntr submission. The report to Huntr was written independently, against `torch 2.6.0` and pytorch `676ae52`, and cites neither issue. No claim of first discovery is made for instance 1. [#179581](https://github.com/pytorch/pytorch/issues/179581) (2026-04-07) had already listed a SEGV in `SourceRangeDeserializer::deserialize_source()` among four fuzzing crashes, without identifying the cause.
- #185441 was closed as not planned on 2026-08-07. [PR #187153](https://github.com/pytorch/pytorch/pull/187153) with a bounds check was closed unmerged on 2026-09-21.
- Instance 2 (storage size versus record size): no public issue, PR or advisory found. A search of pytorch/pytorch issues for the unpickler storage path and record size returned nothing relevant.
- No CVE or GHSA for either instance was found. Existing PyTorch advisories on `torch.load` and RemoteModule concern different code paths.

## References

- Instance 1: [`source_range_serialization.cpp` @ v2.15.0, L70-L85](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/jit/serialization/source_range_serialization.cpp#L70-L85)
- Instance 2: [`import_read.cpp` L40-L43](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/jit/serialization/import_read.cpp#L40-L43), [`unpickler.cpp` L572-L596](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/jit/serialization/unpickler.cpp#L572-L596)
- [PyTorch `SECURITY.md`](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/SECURITY.md)
- Issues [#185441](https://github.com/pytorch/pytorch/issues/185441), [#179581](https://github.com/pytorch/pytorch/issues/179581); PR [#187153](https://github.com/pytorch/pytorch/pull/187153)
- [CWE-125](https://cwe.mitre.org/cgi-bin/jumpmenu.cgi?id=125)

## License

MIT, see [LICENSE](LICENSE).
