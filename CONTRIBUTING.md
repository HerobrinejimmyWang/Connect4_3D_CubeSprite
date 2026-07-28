# Contributing

Thank you for helping improve Connect4 3D CubeSprite.

## Change scope

Keep commits focused on one concern. Source code, tests, manifests, lock files,
release notes, benchmark summaries, and reproducible charts are appropriate for
version control.

Do not commit:

- `.tmp/`, `tmp/`, `tmp_built_app/`, editor settings, or Python caches;
- `node_modules/`, frontend `dist/`, Rust `target/`, or PyInstaller output;
- generated sidecar executables, installers, archives, or benchmark logs;
- large showcase videos;
- ad hoc checkpoints or training caches.

Bundled release models are an intentional exception: the registered ONNX files
under `desktop_app/src-tauri/resources/models/` are tracked with Git LFS and
must match the SHA-256 values in `model_registry.json`.

## Validation

For desktop changes:

```powershell
cd desktop_app
pnpm typecheck
pnpm test
pnpm build
```

For backend and shared Python changes, run from the repository root:

```powershell
python -m unittest discover -s desktop_app\backend\tests -v
python -m unittest discover -s desktop_app\tests_backend -v
python -m compileall connect4_core connect4_runtime arena game_client desktop_app\backend
```

Rebuild the sidecar and installer only when preparing a distributable build.
Generated executables should be attached to a GitHub Release, not committed.

## Bug reports

Include the application version, Windows version, model choice, MCTS
simulations, temperature, and exact steps to reproduce. For gameplay or replay
issues, attach an exported replay when it does not contain sensitive material.
