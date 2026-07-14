"""PyInstaller entry point that preserves the backend package context."""

from cubesprite_backend.main import main


if __name__ == "__main__":
    raise SystemExit(main())
