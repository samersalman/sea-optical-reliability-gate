# Weights license notes

Checked 2026-10-20 against the sources linked below. This file is not legal advice. Read the linked texts before you reuse or redistribute any checkpoint.

## 1. Files covered

The release `v1.0.0` carries 12 PyTorch `state_dict` checkpoint files (SHA-256 values in `v1/config/history/weights_sha256.txt`) and two companion files, `LICENSE-NVIDIA-SegFormer.txt` (section 3) and `SHA256SUMS`. All parameters were trained on SEA Center 1 frames (`v1/python/04_train.py`); nothing is frozen.

The architecture and checkpoint identifiers are not string literals in `models.py`. They are set under `segmenters.models` in `v1/config/analysis_plan.yaml`, and `v1/python/models.py` reads them.

| Files | Network | Started from | Terms |
|---|---|---|---|
| `unet_fold0.pt` to `unet_fold4.pt`, `unet_final.pt` | U-Net, ResNet34 encoder | ImageNet ResNet34 (via `segmentation_models_pytorch`) | MIT for the authors' part; see section 2 |
| `segformer_fold0.pt` to `segformer_fold4.pt`, `segformer_final.pt` | SegFormer-B0 | `nvidia/mit-b0` | NVIDIA Source Code License for SegFormer; see section 3 |

## 2. U-Net weights

- `segmenters.models.unet` in `analysis_plan.yaml` sets `encoder: resnet34` and `weights: imagenet`, and `models.py` passes them to `smp.Unet`. Version 0.5.0 of the library loads the encoder from Hugging Face `smp-hub/resnet34.imagenet` (revision `7a57b34f723329ff020b3f8bc41771163c519d0c`). Its model card gives the origin as "https://download.pytorch.org/models/resnet34-333f7ec4.pth", the torchvision ImageNet ResNet34. The decoder and head start from random values.
- The library is MIT ("Copyright (c) 2019, Pavel Iakubovskii"). torchvision's code is BSD 3-Clause.
- The `smp-hub/resnet34.imagenet` card states `license: other` and gives no license text.
- torchvision states: "The pre-trained models provided in this library may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case."
- ImageNet's terms of access say: "Researcher shall use the Database only for non-commercial research and educational purposes." That text binds a researcher who requests the database. None of the sources above says whether it reaches weights trained on ImageNet or models fine-tuned from them. This is unclear.
- Result: the authors license the U-Net checkpoints under the MIT License (text in `LICENSE`). That grant covers only what the authors can grant, namely their own training on SEA. It does not remove any terms that attach to the ImageNet-pretrained initialization. Check those terms before commercial use.

## 3. SegFormer weights

- `segmenters.models.segformer` in `analysis_plan.yaml` sets `checkpoint: nvidia/mit-b0` and `num_labels: 1`, and `models.py` passes them to `SegformerForSemanticSegmentation.from_pretrained`. The decode head is new. The encoder weights are then trained further.
- The local Hugging Face cache holds two snapshots of `nvidia/mit-b0`: `80983a413c30d36a39c20203974ae7807835e2b4` (`refs/main`; `config.json` and `pytorch_model.bin`) and `25ce79d97e6d9d509ed12e17cb2eb89b0a83a2dc` (`refs/pr/5`; `model.safetensors` only). `models.py` sets `HF_HUB_OFFLINE=1`. With transformers 5.2.0 (`v1/config/history/pip_freeze.txt`), `from_pretrained` resolves `main`, finds no `model.safetensors` there, and loads `pytorch_model.bin` from `80983a41`. A re-run on 2026-10-20 confirmed this; the training run itself was not traced. The two files hold identical tensors (194 of 194 keys equal), so the starting weights are the same either way.
- The `nvidia/mit-b0` card states `license: other` and says: "The license for this model can be found here", linking to the NVlabs LICENSE. GitHub lists that repository's license as "Other" (no SPDX identifier). The text is titled "NVIDIA Source Code License for SegFormer". The copy read on 2026-10-20 has SHA-256 `f549820c06519e3105e5174a2fd7285224ffd2268a7f879a843b1a255fd04a61`.
- Terms that bind a user of these weights:
  - Use (3.3): "The Work and any derivative works thereof only may be used or intended for use non-commercially." and "As used herein, "non-commercially" means for research or evaluation purposes only." NVIDIA and its affiliates keep a commercial exception for themselves.
  - Redistribution (3.1): "(a) you do so under this License, (b) you include a complete copy of this License with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work."
  - Derivative works (3.2): extra terms are allowed only if they "provide that the use limitation in Section 3.3 applies to your derivative works", and the derivative works must be identified.
  - Trademarks (3.5): no right to use NVIDIA's "names, logos, or trademarks", except to reproduce required notices.
  - Termination (3.4, 3.6): rights end immediately on a violation or on a patent claim against a Licensor.
  - Warranty (4) and liability (5): the Work is provided "AS IS".
  - The NVlabs README adds: "SegFormer may be used non-commercially, meaning for research or evaluation purposes only." It points commercial inquiries to https://www.nvidia.com/en-us/research/inquiries/.
- The definition of "derivative works" excludes "works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work." The license does not say whether fine-tuned weights count. The encoder weights here are changed in place and cannot be separated, so this repository treats the six SegFormer checkpoints as derivative works. That is a reading, not a statement by NVIDIA.
- Result: `segformer_fold0.pt` to `segformer_fold4.pt` and `segformer_final.pt` are distributed under the NVIDIA Source Code License for SegFormer, with no additional terms. Use is limited to research or evaluation. The release includes a complete copy of the license as `LICENSE-NVIDIA-SegFormer.txt` next to these files.
- No copyright or attribution line was found in the model card, the LICENSE file, or the safetensors header of `nvidia/mit-b0`, so there is no notice to carry forward. Please cite Xie et al. 2021, arXiv:2105.15203.
- The `mit-b0` card lists ImageNet-1k as its pretraining data, so the ImageNet point in section 2 applies here as well.

## 4. What the repository MIT license covers

`LICENSE` covers the code in this repository and the U-Net checkpoints as described in section 2. It does not cover the SegFormer checkpoints. It does not relicense NVIDIA's work or the ImageNet-pretrained weights. No NVIDIA source code is copied into the repository; `models.py` calls the Hugging Face `transformers` implementation.

## 5. Data note

The weights were trained on SEA (Spine Endoscopic Atlas), Figshare doi:10.6084/m9.figshare.27109312.v1, which the Figshare API lists as CC0 (https://creativecommons.org/publicdomain/zero/1.0/). The CC0 1.0 legal code waives the Affirmer's rights in the Work "for any purpose whatsoever, including without limitation commercial, advertising or promotional purposes" (section 2) and does not mention trained models. The authors read CC0 as adding no condition of its own to the checkpoints; that is their reading, not legal advice. The SEA descriptor article (Xu et al., Sci Data 2025, doi:10.1038/s41597-025-05897-7) has its own article license, which does not govern the data.

## 6. Sources (accessed 2026-10-20)

- https://huggingface.co/nvidia/mit-b0
- https://github.com/NVlabs/SegFormer/blob/master/LICENSE and https://github.com/NVlabs/SegFormer#license
- https://huggingface.co/smp-hub/resnet34.imagenet
- https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE
- https://github.com/pytorch/vision#pre-trained-model-license and https://github.com/pytorch/vision/blob/main/LICENSE
- https://www.image-net.org/download.php
- https://api.figshare.com/v2/articles/27109312
- https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt

Not legal advice. If a point matters for your use, read the linked texts or ask the licensor.
