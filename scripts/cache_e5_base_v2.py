from __future__ import annotations

from pathlib import Path


MODEL_ID = "intfloat/e5-base-v2"
TARGET_DIR = Path(__file__).resolve().parents[1] / "models" / "intfloat" / "e5-base-v2"


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install dense dependencies first: uv sync --extra dense --extra yaml --group dev") from exc

    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_ID)
    model.save_pretrained(str(TARGET_DIR))

    SentenceTransformer(str(TARGET_DIR), local_files_only=True)
    print(f"Stored {MODEL_ID} in {TARGET_DIR}")


if __name__ == "__main__":
    main()
