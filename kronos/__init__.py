from .inference import create_model, create_model_from_pretrained
from .preprocessing import (
	MarkerChannelMetadata,
	crop_patch,
	default_marker_ids,
	infer_intensity_max_value,
	load_marker_info_with_metadata_csv,
	marker_ids_from_names,
	prepare_kronos_input,
)

from .qptiff_model import KronosForQPTIFFCoordinateRandomDataset


def create_qptiff_model(
	*,
	cfg: dict,
	in_markers: int = 8,
	out_channels: int = 1,
	decoder_dim: int = 256,
	adapter_mlp_dim: int = 128,
	adapter_conv_kernel: int = 3,
	return_embeddings: bool = False,
):
	"""Create a KRONOS backbone (no pretrained weights) wrapped for QPTIFF datasets."""

	backbone, precision, embedding_dim = create_model(cfg=cfg)
	model = KronosForQPTIFFCoordinateRandomDataset(
		backbone,
		in_markers=in_markers,
		out_channels=out_channels,
		decoder_dim=decoder_dim,
		adapter_mlp_dim=adapter_mlp_dim,
		adapter_conv_kernel=adapter_conv_kernel,
		return_embeddings=return_embeddings,
	)
	return model, precision, embedding_dim


def create_qptiff_model_from_pretrained(
	*,
	checkpoint_path: str,
	cfg_path: str | None = None,
	cache_dir: str | None = None,
	hf_auth_token: str | None = None,
	cfg: dict | None = None,
	in_markers: int = 8,
	out_channels: int = 1,
	decoder_dim: int = 256,
	adapter_mlp_dim: int = 128,
	adapter_conv_kernel: int = 3,
	return_embeddings: bool = False,
):
	"""Create a pretrained KRONOS backbone wrapped for QPTIFF datasets."""

	backbone, precision, embedding_dim = create_model_from_pretrained(
		checkpoint_path=checkpoint_path,
		cfg_path=cfg_path,
		cache_dir=cache_dir,
		hf_auth_token=hf_auth_token,
		cfg=cfg,
	)
	model = KronosForQPTIFFCoordinateRandomDataset(
		backbone,
		in_markers=in_markers,
		out_channels=out_channels,
		decoder_dim=decoder_dim,
		adapter_mlp_dim=adapter_mlp_dim,
		adapter_conv_kernel=adapter_conv_kernel,
		return_embeddings=return_embeddings,
	)
	return model, precision, embedding_dim


__all__ = [
	"create_model",
	"create_model_from_pretrained",
	"KronosForQPTIFFCoordinateRandomDataset",
	"create_qptiff_model",
	"create_qptiff_model_from_pretrained",
	"MarkerChannelMetadata",
	"crop_patch",
	"default_marker_ids",
	"infer_intensity_max_value",
	"load_marker_info_with_metadata_csv",
	"marker_ids_from_names",
	"prepare_kronos_input",
] 