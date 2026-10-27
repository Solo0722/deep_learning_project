# Local pilot verification

Executed on Windows CPU using synthetic fixtures. No CCAST job or road-damage benefit experiment was run.

- Six tests passed; see test_output.txt.
- Independently executed identity branches matched model parameters, batch-normalization buffers, optimizer, scheduler, RNG state, first gradient/update, predictions, and metrics exactly. Maximum model difference: 0.
- A paused/resumed trajectory matched uninterrupted training state exactly.
- Identity probe gains were zero; a nonidentity brightness intervention changed only its target class.
- Group isolation, transitive links, design tampering, crop checksum rejection, unchanged padding, zero magnitude, and deterministic paired batches were checked.
- Class-balanced loss matched a known analytic case; its gradient matched finite differences under unequal class support.
- Original-PNG preview generation was exercised with synthetic training images and absent test pixels.
- The real frozen manifest yielded reference/audit role sizes of 4116/4126; each selected panel has 128 crops per class.
- 35 real-data proxy preview sheets were generated; five representative sheets were visually inspected, covering all operations.
- Baseline source and manuscript/bibliography SHA256 values match the pre-work values.

These checks establish local software behavior, not CUDA determinism, class-label preservation, effectiveness of augmentation, independent replication, or novelty. The GPU identity command in PILOT_RUNBOOK.md is still required on CCAST. Strength ranges, horizons, panels, and seed counts remain pilot choices.

## Provenance

```json
{
  "code_hashes": {
    "pilot_runner.py": "2474469d3b86b73cd18edf7d1d33f2ad90d00ca1a0434ef878d4b1701df3dc71",
    "train_baseline.py": "594cbfe8dfd955697779dd1b0af215135f8e71b77bcae1df4570c730201d6d55"
  },
  "design_sha256": "026a28ee9dd219d12f946cf50134af8a9d8d4f08aec04fcd447ed57199b76af4",
  "manifest_sha256": "6c319939a0b913693935c5e6e83acb8e12b442c84e13fef3d79b740423a01e78",
  "preserved_files": {
    "train_baseline.py": "594cbfe8dfd955697779dd1b0af215135f8e71b77bcae1df4570c730201d6d55",
    "research_paper/main.tex": "13806ecda86b39c9558d6c479f54c89f8fa0038d81bd2066b4f80023dbd1bd96",
    "research_paper/references.bib": "dd04458e24817c8a6d014470cf56cd2d4e876611a61fe883751ffb9fc6ff9a93"
  },
  "environment": {
    "utc": "2026-10-16T07:51:10.010451+00:00",
    "python": "3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)]",
    "torch": "2.10.0+cpu",
    "numpy": "2.4.2",
    "pillow": "12.1.1",
    "platform": "Windows-11-10.0.26300-SP0",
    "host": "DevEx-Alchemy",
    "device": "cpu",
    "gpu": null,
    "cuda_build": null,
    "argv": [
      "python.exe -m unittest",
      "discover",
      "-s",
      "tests",
      "-p",
      "test_pilot_runner.py",
      "-v"
    ]
  }
}
```
