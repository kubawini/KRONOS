import torch
import logging

from kronos import create_qptiff_model_from_pretrained

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Loads pretrained KRONOS weights for the backbone.
    # Note: the dense decoder head in the wrapper is randomly initialized
    # until you train it for your reconstruction/segmentation objective.
    model, precision, embedding_dim = create_qptiff_model_from_pretrained(
        checkpoint_path="hf_hub:MahmoodLab/kronos",
        cache_dir="./model_assets",
        cfg={"model_type": "vits16", "token_overlap": False},
        in_markers=8,
        out_channels=1,
        return_embeddings=False,
    )

    print("Backbone embedding dim:", embedding_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    batch_size = 2
    marker_count = 8
    window_size = 224

    # Dataset-style batch
    sample = {
        "image": torch.rand(batch_size, marker_count, window_size, window_size, dtype=precision, device=device),
        "mask": torch.rand(batch_size, 1, window_size, window_size, dtype=precision, device=device),
        "folder": ["example_a", "example_b"],
        "idx": torch.tensor([0, 1], device=device),
    }

    model.eval()
    with torch.inference_mode():
        out = model(sample)

    print("Pred mask:", tuple(out.pred_mask.shape))
    print("Target mask:", None if out.target_mask is None else tuple(out.target_mask.shape))
