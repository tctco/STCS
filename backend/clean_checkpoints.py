import shutil
import torch
from pathlib import Path

SRC_ROOT = Path("/backend/trained_models")
DST_ROOT = Path("/backend/trained_models_clean")

DST_ROOT.mkdir(parents=True, exist_ok=True)


def extract_state_dict(ckpt: dict):
    """
    尝试从 OpenMMLab / PyTorch checkpoint 中提取纯权重
    """
    if isinstance(ckpt, dict):
        # 常见 OpenMMLab / mmengine
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]

    # 已经是纯 state_dict（参数名 -> tensor）
    if isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        return ckpt

    raise ValueError("Unrecognized checkpoint format (cannot extract state_dict)")


def clean_pth(src: Path, dst: Path):
    print(f"[PTH] cleaning: {src}")
    try:
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        state = extract_state_dict(ckpt)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, dst)
        print(f"      -> saved weights-only to {dst}")
    except Exception as e:
        print(f"      !! failed to clean {src}: {e}")
        raise


def copy_other(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}")


def main():
    for src in SRC_ROOT.rglob("*"):
        if not src.is_file():
            continue

        rel = src.relative_to(SRC_ROOT)
        dst = DST_ROOT / rel

        if src.suffix == ".pth":
            clean_pth(src, dst)
        else:
            copy_other(src, dst)


if __name__ == "__main__":
    main()
