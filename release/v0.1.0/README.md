# Connect4 3D CubeSprite v0.1.0

Status: **release materials in preparation**

This directory collects the introduction, model notes, presentation, benchmark
evidence, and showcase sources for the first semantic-versioned CubeSprite
desktop release.

## Contents

- `introducing.txt` — author's working introduction and change summary.
- `MODEL_TRAINING_HISTORY.md` — author-owned model lineage and training history.
- `demo.pptx` — presentation source.
- `demo/` — local showcase clips; ignored by Git and intended for release-asset
  hosting.
- `test_results/cpu_response_latency/` — per-model CPU latency evidence.
- `test_results/mcts_scaling/` — MCTS scaling data and chart.

## Planned GitHub Release assets

| Artifact | Distribution | SHA-256 |
| --- | --- | --- |
| `Connect4 3D CubeSprite_0.1.0_x64-setup.exe` | Windows desktop installer | `d81e6870f7eeef187d1e9984224663e980710b90a85dc734d86a13aafc317901` |
| `Connect4_3D_CubeSprite_0.1.0-beta-android-arm64.apk` | **Android ARM64 Beta** test build | `708b89c85de090ea8701db62f371974f83be57194087f5a218ed650f508bb038` |
| `Connect4_3D_CubeSprite_models_v0.1.0.zip` | Four weights-only PyTorch `.pth` state dictionaries | `873430fc0d9df6a90bb3ccc7a77145aa2b9fc6df476d85b1add19aa419c4ddea` |

Do not use a hash from an intermediate build. Record the installer hash only
after the final source commit, model registry, bundled models, version number,
and release notes have been frozen. The Android APK must remain clearly marked
as a Beta test build in both its filename and GitHub Release description.

## Release checklist

- [ ] Finish and review the model training history.
- [ ] Finish the showcase video and presentation.
- [x] Confirm the four bundled model names, iterations, and SHA-256 hashes.
- [x] Package the four weights-only `.pth` state dictionaries without optimizer,
      trainer, evaluation, or other checkpoint state.
- [ ] Review model provenance and third-party license notices.
- [x] Run frontend type checking and the complete frontend test suite.
- [ ] Run both backend test suites.
- [ ] Build the Python sidecar from a clean environment.
- [ ] Build and install the NSIS package on a clean Windows account or machine.
- [x] Build the Android Beta APK and smoke-test it on a supported Android device.
- [ ] Smoke-test new game, settings, AI settings, hints, 3D view, and replay.
- [ ] Confirm the application works without a development environment.
- [x] Record final installer size and SHA-256.
- [ ] Create the `v0.1.0` Git tag from the exact release commit.
- [ ] Upload the Windows installer, Android Beta APK, and model-weight archive
      to the GitHub Release.

## Known release boundary

This release publishes the verified CubeSprite game client and desktop
application. The historical training pipeline remains available on the
`legacy-training` branch and is not presented as fully end-to-end validated.
