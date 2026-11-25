# Exact finite-FP8 matrix multiplication on Ascend 910C

**Technical validation report | v1.0.0 | 2026-11-25**  
**Status:** hardware-tested correctness-first prototype (host/NPU hybrid)

## Abstract

This project implements exact mathematical matrix multiplication of finite FP8 E4M3FN and E5M2 values without native FP8 Cube instructions. Raw FP8 values are mapped to exact signed integers, decomposed into radix-128 signed INT8 digit planes, multiplied on the Ascend 910C Cube using INT32 partial accumulators, and reconstructed on the host with signed 128-bit integer arithmetic. No lossy quantization or floating-point accumulation is required. The supplied Device 0 log for CANN 9.0.0 reports 64 successful matrix cases, 111,149,056 checked partial outputs and 65,536 checked final coefficients, with zero mismatches. The result establishes correctness for the logged test cases and does not establish native FP8 support, full on-device execution or peak throughput.

## 1. Scope and evidence

The release preserves `fp8_matmul.asc`, `main.cpp`, `config.h`, `CMakeLists.txt`, `run.sh` and `cpu_test.sh` byte-for-byte from link-fix v2. Their hashes are archived in [tested_source.sha256](../results/tested_source.sha256). The hardware log supplied by the operator is archived [here](../results/npu_device0_2026-11-25.log), with only the terminal hostname replaced and line endings normalized. The association between these source files and the hardware log follows the reported test workflow; the log itself does not embed a source hash. No new hardware run was performed during packaging. Two CPU-only tests were rerun, and the archived NPU log was parsed and audited. Earlier K=33 hardware smoke results are not included in the aggregate below because the archived multi-K log starts at K=256.

## 2. Mathematical formulation

Let S, E and M denote sign, exponent field and mantissa, and m denote the number of mantissa bits. The finite FP8 value x has an exact common-denominator representation:

- E4M3FN: m=3, q=9, x=I·2^(-9), |I| ≤ 229,376.
- E5M2: m=2, q=16, x=I·2^(-16), |I| ≤ 3,758,096,384.

For E=0, I=(-1)^S M. Otherwise I=(-1)^S (2^m+M)·2^(E-1). E4M3FN rejects E=15,M=7, and E5M2 rejects E=31. The largest finite E4M3FN value is 448; for E5M2 it is 57,344. All nonzero subnormal values remain exact under the integer mapping. Signed zero is represented by integer zero.

Split I into signed radix-128 digits: I=Σ d_i·2^(7i), with -127 ≤ d_i ≤ 127. Each digit fits in native signed INT8. The decomposition reconstructs the integer exactly, so it is a representation change, not approximate quantization. Three digit planes suffice for E4M3FN and five for E5M2.

For a pair of matrices A and B, let A_i and B_j be digit planes and let t identify the K tile. The exact result coefficient is

```
C_int = Σ_t Σ_i Σ_j (A_i,t @ B_j,t) · 2^(7(i+j))
C = C_int · 2^(-(q_A + q_B))
```

This follows by distributivity over integer addition and multiplication. The kernel's INT32 products are exact provided no overflow or unintended conversion occurs. For TK=32 the conservative magnitude bound is 516,128, below 2^31−1. The host multiplies signed partial values by positive powers of two rather than left-shifting negative signed integers. The reconstruction has sufficient width for all supported K, including cancellation: since all digits of each input share its sign, the sum of absolute weighted contributions is bounded by K·max|I_A|·max|I_B|.

| Format pair | Planes A×B | Common exponent | Max abs coefficient at K=8192 | Signed bits |
|---|---:|---:|---:|---:|
| E4M3FN×E4M3FN | 3×3 | -18 | 431,008,558,088,192 | 50 |
| Mixed, either orientation | 3×5 or 5×3 | -25 | 7,061,644,215,716,937,728 | 64 |
| E5M2×E5M2 | 5×5 | -32 | 115,697,978,830,306,307,735,552 | 78 |

The final representation retains all mathematical information for finite numerical values within the supported shape. Converting it to FP32 may round, and reproducing the rounded accumulation of another device is a different objective. NaN/Inf semantics and negative-zero sign preservation are outside this contract.

## 3. Implementation

| Stage | Processor | Representation / API |
|---|---|---|
| Input decoding | CPU | raw UINT8 to exact signed digits |
| Tile packing | CPU | row-major signed INT8; zero-pad K tail |
| Layout and loading | NPU AIC | ND-to-NZ `DataCopy`, `LoadData`, `LoadDataWithTranspose` |
| Matrix multiplication | NPU AIC | `Mmad`, INT8 × INT8 → INT32 |
| Tile output | NPU AIC | `Fixpipe`, `NoQuant`, ReLU disabled |
| Reconstruction | CPU | weighted signed `__int128` sums |
| Verification | CPU | every partial output and exact integer dot product |

The logical output shape is M=16, N=64. TK=32 is fixed. A maximum of 20 independent jobs can be launched at once, with separate GM output ranges. Each K tile requires 9, 15 or 25 digit-pair products. The host checks each partial against an explicit INT32 dot product and the final coefficients against an independent mathematical FP8 decode and 128-bit dot product. The decoder and reference share the finite-code classifier, a limitation of independence. The reference uses double only for individual exactly representable decoded values and powers-of-two scaling, never for accumulation.

## 4. Test setup and methodology

- Reported target: Ascend 910C, Device 0; compiler command specifies `dav-2201` and CANN 9.0.0 Bisheng. CMake 3.20.5 appears in the supplied log. Linking includes the corrected `libascendc_runtime.a` and dependencies.
- Each run: K=256, 1024, 4096 or 8192; one round; maximum 20 Cube blocks per launch. All four ordered format pairs and four patterns are evaluated (16 cases per K).
- Pattern 0: pseudo-random finite encodings, seed 910. Pattern 1: all largest positive finite values. Pattern 2: repeating large positive, tiny and large negative products. Pattern 3: positive and negative zero inputs. Decoding tests cover all 256 encodings per format, including rejection of 2 E4M3FN and 8 E5M2 special encodings.
- Pass condition: zero partial and final mismatches; no numeric tolerance. The output buffer is prefilled with a poison value before launch to help expose missing writes. The audit separately checks recorded counts, exponents and printed deterministic coefficients.

## 5. Hardware results

| K | Cases | Partial outputs checked | Final outputs checked | Mismatches |
|---:|---:|---:|---:|---:|
| 256 | 16 | 2,097,152 | 16,384 | 0 |
| 1024 | 16 | 8,388,608 | 16,384 | 0 |
| 4096 | 16 | 33,554,432 | 16,384 | 0 |
| 8192 | 16 | 67,108,864 | 16,384 | 0 |
| **Total** | **64** | **111,149,056** | **65,536** | **0** |

Each run ended with `FINAL: PASS (all checked integer results exact)`. The tile count is `ceil(K/32) × planes_A × planes_B`, and each tile contributes 1024 partial outputs. For K=8192, E5M2×E5M2 checks 6,553,600 partial outputs per case. Its maximum-positive sample is exactly `115697978830306307735552 * 2^(-32)`, matching the independent worst-case bound. Its mixed-sign cancellation sample is `14123288431433878187 * 2^(-32)`. Both signed-zero patterns reconstruct zero.

See [CSV](../results/npu_cases.csv) for every recorded case and [audit output](../results/npu_audit.txt) for independent log consistency checks. These are counts of checked outputs, not counts of independent hardware trials or a statistical reliability guarantee.

## 6. Timing analysis

Medians across the four different patterns at each K, measured in milliseconds. These are descriptive summaries, not repeated-identical-input benchmark medians.

| K | E4×E4 transfer/launch | E4×E5 transfer/launch | E5×E4 transfer/launch | E5×E5 transfer/launch |
|---:|---:|---:|---:|---:|
| 256 | 0.3350 | 0.4605 | 0.4550 | 0.7560 |
| 1024 | 1.1875 | 1.8625 | 1.8590 | 3.0210 |
| 4096 | 4.4065 | 7.2830 | 7.2430 | 11.9690 |
| 8192 | 8.7855 | 14.3750 | 14.5450 | 24.1490 |

At K=8192, median checked wall times are 533.827, 830.1385, 830.1395 and 1324.311 ms respectively. Only roughly 1.6–1.8% of checked wall time lies inside `Engine::run`. The remainder includes host tile packing, partial reference calculations, reconstruction and final checks; the log does not isolate those subcomponents. Initial input creation and decode are outside checked wall time. Engine time includes copies of A, B and poisoned C, kernel launch, synchronization and C readback. It is not a device-event kernel duration.

For illustration, the logical work convention 2MNK gives 16,777,216 operations at K=8192. Dividing by the measured engine times gives approximately 1.910, 1.167, 1.154 and 0.695 logical GOPS. This is not native FP8 throughput, and excludes CPU decode, reconstruction and verification. No peak TOPS estimate or comparison with GPU vendor performance is justified by this test.

The small M/N output and K tile of 32 generate many tiny jobs. At K=8192 the E4×E4 case returns 9 MiB of partial INT32 data; E5×E5 returns 25 MiB, before CPU reconstruction. Repeated transfers and per-tile output are obvious targets for future optimization.

## 7. Reproducibility and release checks

From the repository root:

```bash
sha256sum -c results/tested_source.sha256
python3 scripts/audit_log.py
bash cpu_test.sh 33 1 7
bash cpu_test.sh 8192 1 20
# On the target NPU only:
for k in 256 1024 4096 8192; do bash run.sh 0 "$k" 1 20 || exit 1; done
```

The release packaging CPU runs at K=33 and K=8192 both report PASS; see [small](../results/release_cpu_k33.log) and [long](../results/release_cpu_k8192.log) logs. GitHub Actions repeats CPU checks and validates the archive; it does not run on Ascend hardware. The archived log audit validates consistency of the recorded results, not independent hardware correctness for unprinted data.

## 8. Limitations and next steps

This is a fixed-shape demonstration, not a drop-in FP8 GEMM library. It rejects NaN and infinity, normalizes signed zeros, does not support external scaling factors, and exposes only a sample output plus mismatch counts. Device-side decode and wide reconstruction have not been implemented. Only Device 0 is documented in the archived runs; actual Cube utilization, long-duration repeatability, maximum M/N, device-1 behavior, dual-chip performance and peak throughput are not established.

Future work should preserve the exact output contract while reducing the number of host transfers: accumulate K tiles on Cube where the INT32 bound permits; use AIV LUTs for exact decomposition; implement explicit carry propagation for wide device-side reconstruction; add dynamic M/N tiling; separate verification and throughput timing; validate adversarial cases, long loops and both devices. Every optimized path needs new hardware evidence. These are proposed directions, not v1.0.0 features.

## 9. Conclusion

The archived results support the feasibility of exact finite-FP8 matrix arithmetic on Ascend 910C by combining exact integer representation, native INT8 Cube tile products and host-side wide accumulation. All 64 archived test cases report zero mismatches. The key contribution is the arithmetic contract and reproducible correctness baseline, not a claim of native FP8 or competitive throughput. Preserve the source and logs as a baseline for future optimization.

## References

See the official CANN API and OFP8 specification links in the [README](../README.md). The algorithm is an independent implementation, not an official Huawei solution.
