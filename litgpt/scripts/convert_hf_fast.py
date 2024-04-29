import torch
from pathlib import Path


@torch.inference_mode()
def convert_lit_checkpoint(checkpoint_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pytorch_model.bin"
    ckpt = torch.load(checkpoint_dir, map_location="cpu")
    torch.save(ckpt["module"], output_path)
    


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint)
