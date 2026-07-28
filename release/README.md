# Release materials

This directory contains versioned source material used to prepare public
Connect4 3D CubeSprite releases.

## Version directory convention

Each `vX.Y.Z/` directory may contain:

- release notes and an introduction;
- a model training history;
- reproducible benchmark summaries and machine-readable results;
- charts and presentation sources;
- a checklist recording what was validated before publication.

Large media and generated packages are intentionally not committed:

- showcase videos (`.mp4`, `.mov`, `.mkv`, `.webm`);
- installers and archives (`.exe`, `.msi`, `.zip`, `.7z`);
- benchmark runner logs;
- temporary Office lock files.

Upload those files directly as GitHub Release assets or to the selected external
distribution service. Keep hashes for distributed installers in the
corresponding version README.

Current prepared release: [v0.1.0](v0.1.0/README.md).
