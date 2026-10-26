"""Preserve the source/target supervision protocol used by the merged model."""

from __future__ import annotations


_PATCHED_TOKEN_IDS: set[int] = set()


def patch_source_token_loss(source_token_id: int) -> None:
    """Mask supervised SOURCE_TOKEN positions in LlamaFactory SFT labels.

    The original training protocol supervises the source transcript, target
    marker, translation and EOS, but not the leading source marker itself.
    Patching the common encoder also covers the packed processor because it
    inherits this method.
    """
    if source_token_id in _PATCHED_TOKEN_IDS:
        return

    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

    original = SupervisedDatasetProcessor._encode_data_example

    def _encode_and_mask(self, *args, **kwargs):
        input_ids, labels = original(self, *args, **kwargs)
        labels = [
            -100 if token_id == source_token_id and label != -100 else label
            for token_id, label in zip(input_ids, labels, strict=True)
        ]
        return input_ids, labels

    SupervisedDatasetProcessor._encode_data_example = _encode_and_mask
    _PATCHED_TOKEN_IDS.add(source_token_id)


__all__ = ["patch_source_token_loss"]
