# baseline-onion-gpu

A faithful port of [NIKIB0Y/onion-gpu](https://github.com/NIKIB0Y/onion-gpu)
(MIT, commit `24e5874`, 2026-09-27) to the `onionrig-grind/1` contract
(`docs/design/CONTRACT.md`). One source builds for CUDA, to test locally on
NVIDIA, and for HIP (gfx90a, gfx950).

This is the **calibration floor**, not a competitor. It shows what a faithful
port of the best-documented NVIDIA kernel delivers on CDNA without any AMD
tuning, and every entry is measured against it.

## Provenance

- `src/fe25519.cuh` and `src/ge25519.cuh` are byte-for-byte upstream. Check with
  `cmp src/fe25519.cuh upstream/fe25519.cuh`.
- `upstream/` holds the original `main.cu`, the headers, the README, and the exact
  commit hash in `upstream/COMMIT`.
- `LICENSE.upstream` is the original MIT license (Copyright (c) 2026 Niklas Schillitz).

## What changed, and why

| Change | Why |
|---|---|
| Each thread starts at `A0 + k_base*(8B)` from the rig's public `A0`, instead of a secret scalar derived on the GPU | Split-key search: no secret material ever reaches the device. As a bonus, SHA-512, SHA3, nonce generation, and key-file writing all drop out of the entry. |
| Hits report the signed offset `k` and support up to 256 targets, including targets longer than 12 characters | Contract. The 64-bit fast-path compare and the cold-path full check are kept. |
| New `ge_extra.cuh` adds RFC 8032 decoding of `A0`, negation, and signed offsets | Needed to start from a public point. It is built only from upstream primitives. |
| Host code speaks `--info`, `--verify`, and `--grind` | Contract. |
| Launch geometry, `BINV_N`, `INNER_BATCH`, `run_batch()`, and the differential self-check are unchanged | They define the baseline. |

**A latent quirk in upstream.** Upstream's `fe_frombytes` does not simply
ignore bit 255, which RFC 8032 requires; it applies an unusual correction. Upstream
only ever passes it constants with that bit clear, so the quirk never shows. In
a point encoding, bit 255 is the x sign, so `ge_frombytes` clears it first. The
shared decode, offset, and known-answer vectors caught this on the first GPU run:
without the fix, roughly half of all start points decoded wrong.

## Build

    make cuda                    # NVIDIA, for local testing
    make hip                     # gfx90a + gfx950 fat binary (needs ROCm)
    make resource-usage          # per-kernel VGPRs / scratch / occupancy per AMD arch
    make isa                     # AMD assembly per arch

Without a local ROCm install, use the official image:

    docker run --rm -v "$PWD":/w -w /w rocm/dev-ubuntu-24.04:7.2.4 make hip resource-usage

## Measured and static results

**NVIDIA RTX 4060 Ti (sm_89, CUDA 13.4).** The rig's proof-of-work measurement
gives about 918 M candidates/s. Upstream reports 917 M keys/s on an RTX 5060 for
the same kernel, so the port costs nothing.

**AMD (static, from the ROCm 7.2.4 compiler).** `grind_kernel` resource usage:

| | gfx90a (MI210) | gfx950 (MI355X) |
|---|---|---|
| VGPRs / AGPRs | 256 / 6 | 256 / 14 |
| Scratch per lane | 31,340 B | 31,316 B |
| SGPR spills | 143 | 145 |
| Occupancy | **1 wave/SIMD** | **1 wave/SIMD** |

On NVIDIA, upstream's launch bounds cap registers at about 170 per thread, and
the ref10 int64-accumulator multiply fits because NVIDIA has a native wide
multiply-add. On CDNA every 64-bit value costs two VGPRs, so the compiler uses all
256 arch VGPRs, spills into AccVGPRs, and leaves one wave per SIMD, with nothing
to hide the latency of the ~30 KB batch arrays in scratch. Register pressure and
occupancy are the obvious first lever for any AMD entry, which matches the GPU
MODE craft notes in `docs/research/gpumode-craft.md`.
