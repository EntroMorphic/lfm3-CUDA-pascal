# native/

The engine. C++20 and CUDA, built for `sm_110` with `--fmad=false` (see [docs/building.md](../docs/building.md)).

| Path | Contents |
|---|---|
| `include/lfm/` | The engine's headers: text front end, model operators, GEMM, encoders, depth decoder, codec, HTTP layer |
| `src/` | Executables and their translation units: `lfm3_server.cu` (`liquid-lfm3-server`), `lfm3_worker.cu`, `lfm3_main.cu` (`liquid-lfm3`, and the `Runner` every other target includes), image decoding and preprocessing |
| `apps/` | Driver CLIs that produce results for validation rounds: `lfm3-speech` (text, audio and image in; text and speech out), `lfm3-batch`, `lfm3-tree`, `lfm3-score` |
| `tests/` | Pass/fail checks and oracles, and the host tools the Python tests in `../tests/` drive |
| `bench/` | Benchmarks and parameter sweeps (speed only; never acceptance) |
| `probes/` | Diagnostics, library-behavior studies and parked candidates (no acceptance implied) |
| `third_party/` | Vendored libjpeg-turbo and stb_image, and the licenses of ported code ([README](third_party/README.md)) |
| `CMakeLists.txt` | The server, worker, text engine, codec and depth targets, and the operator test bridge |
| `build.sh`, `profile.sh` | The inherited `liquid-native` (LFM2.5) runtime's build and profile scripts |

Files moved here from `native/tests/` and `native/tools/` in the 2026-11 reorganization; records under
`validation/` cite the old paths ([docs/moved-paths.md](../docs/moved-paths.md) maps them).
