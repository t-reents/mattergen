# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Literal

import fire
from pymatgen.core.structure import Structure

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME, MatterGenCheckpointInfo, ProgressCallback
from mattergen.common.utils.eval_utils import load_structures
from mattergen.generator import CrystalGenerator


def main(
    output_path: str,
    pretrained_name: PRETRAINED_MODEL_NAME | None = None,
    model_path: str | None = None,
    batch_size: int = 64,
    num_batches: int = 1,
    config_overrides: list[str] | None = None,
    checkpoint_epoch: Literal["best", "last"] | int = "last",
    properties_to_condition_on: TargetProperty | None = None,
    sampling_config_path: str | None = None,
    sampling_config_name: str = "default",
    sampling_config_overrides: list[str] | None = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: float | None = None,
    strict_checkpoint_loading: bool = True,
    target_compositions: list[dict[str, int]] | None = None,
    structures_path: str | None = None,
    skip_steps: int = 0,
    num_samples_per_structure: int = 1,
    num_denoising_steps: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[Structure]:
    """
    Evaluate diffusion model against molecular metrics.

    Args:
        model_path: Path to DiffusionLightningModule checkpoint directory.
        output_path: Path to output directory.
        config_overrides: Overrides for the model config, e.g., `model.num_layers=3 model.hidden_dim=128`.
        properties_to_condition_on: Property value to draw conditional sampling with respect to. When this value is an empty dictionary (default), unconditional samples are drawn.
        sampling_config_path: Path to the sampling config file. (default: None, in which case we use `DEFAULT_SAMPLING_CONFIG_PATH` from explorers.common.utils.utils.py)
        sampling_config_name: Name of the sampling config (corresponds to `{sampling_config_path}/{sampling_config_name}.yaml` on disk). (default: default)
        sampling_config_overrides: Overrides for the sampling config, e.g., `condition_loader_partial.batch_size=32`.
        load_epoch: Epoch to load from the checkpoint. If None, the best epoch is loaded. (default: None)
        record: Whether to record the trajectories of the generated structures. (default: True)
        strict_checkpoint_loading: Whether to raise an exception when not all parameters from the checkpoint can be matched to the model.
        target_compositions: List of dictionaries with target compositions to condition on. Each dictionary should have the form `{element: number_of_atoms}`. If None, the target compositions are not conditioned on.
           Only supported for models trained for crystal structure prediction (CSP) (default: None)
        structures_path: Path to starting structures (an .xyz/.extxyz file, a .zip of CIF files, or a directory of CIF/xyz files). If provided, the denoising process starts from these
           structures noised to step `skip_steps` of the noise schedule (SDEdit-style) instead of from pure noise. `num_batches` is ignored in this mode; the number of generated samples is
           len(structures) * num_samples_per_structure, in input order. Only fields with a predictor/corrector in the sampling config are noised and denoised: with the default sampling
           config, positions, lattice and atom types are all noised and denoised; with `--sampling-config-name=pos_only`, the lattice and atom types are kept exactly fixed and only the
           positions are denoised. (default: None)
        skip_steps: Number of leading denoising steps to skip. The noise schedule is unchanged; denoising starts at step `skip_steps` of the N-step grid (e.g., 100 skips the first 100 of
           1000 steps, starting from a lower noise level). Requires structures_path. (default: 0)
        num_samples_per_structure: Number of samples to generate per starting structure. Only used with structures_path. (default: 1)
        num_denoising_steps: Number of denoising steps / noise levels (`N` in the sampling config). If None, the value from the sampling config is used (1000 for the shipped configs).
           Fewer steps means faster but coarser sampling. The number of noise levels of the discrete (D3PM) atom-type corruption is baked into the model config and must match, so this
           also overrides the model config accordingly; no checkpoint parameters are affected. (default: None)
        progress_callback: Optional callback function that takes in a single float argument representing the progress of the generation process (between 0 and 1).
    NOTE: When specifying dictionary values via the CLI, make sure there is no whitespace between the key and value, e.g., `--properties_to_condition_on={key1:value1}`.
    """
    assert (
        pretrained_name is not None or model_path is not None
    ), "Either pretrained_name or model_path must be provided."
    assert (
        pretrained_name is None or model_path is None
    ), "Only one of pretrained_name or model_path can be provided."

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    # Disable generating element types which are not supported or not in the desired chemical
    # system (if provided).
    config_overrides += [
        "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]
    properties_to_condition_on = properties_to_condition_on or {}
    target_compositions = target_compositions or []

    starting_structures: list[Structure] | None = None
    if structures_path is not None:
        assert not target_compositions, "structures_path and target_compositions are mutually exclusive."
        starting_structures = list(load_structures(Path(structures_path)))
        assert len(starting_structures) > 0, f"No structures found at {structures_path}"
    assert skip_steps == 0 or structures_path is not None, "skip_steps > 0 requires structures_path."

    if pretrained_name is not None:
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
            pretrained_name, config_overrides=config_overrides
        )
    else:
        checkpoint_info = MatterGenCheckpointInfo(
            model_path=Path(model_path).resolve(),
            load_epoch=checkpoint_epoch,
            config_overrides=config_overrides,
            strict_checkpoint_loading=strict_checkpoint_loading,
        )
    _sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        diffusion_guidance_factor=(
            diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
        ),
        target_compositions_dict=target_compositions,
        starting_structures=starting_structures,
        num_samples_per_structure=num_samples_per_structure,
        skip_steps=skip_steps,
        num_denoising_steps=num_denoising_steps,
        progress_callback=progress_callback,
    )
    return generator.generate(output_dir=Path(output_path))


def _main():
    # use fire instead of argparse to allow for the specification of dictionary values via the CLI
    fire.Fire(main)


if __name__ == "__main__":
    _main()
