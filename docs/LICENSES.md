# Лицензии и источники / Licenses and data sources

Original team-authored software in the current hackathon release is
**AGPL-3.0-only**, following the owner's explicit attestation of all three
copyright holders' agreement on 11 October 2026. See [LICENSE](../LICENSE),
[NOTICE](../NOTICE) and [the consent record](../.agents/CONTRIBUTOR_LICENSE_CONSENT.md).
Third-party works retain their original licenses; this is not a grant over them.
The final public source/application release is pending exact technical PASS and
verified corresponding-source publication. Historical refusal remains preserved.

## Components and models

| Component / model / data | Version | Source | License | Notes |
| --- | --- | --- | --- | --- |
| Ultralytics (YOLO11n + ByteTrack) | 8.4.175 tested | pypi.org/project/ultralytics | AGPL-3.0 | Detection + tracking. Upstream AGPL obligations must be resolved for the distributed/networked release; public source alone is not a claim of full compliance. See the upstream license below. |
| yolo11n.pt weights | — | github.com/ultralytics/assets | AGPL-3.0 | Pretrained on COCO. Downloaded at first run, not committed. |
| PyTorch (CPU) / TorchVision | 2.14.1+cpu / 0.29.1+cpu tested | pytorch.org | BSD-3-Clause code | Frozen encoder and actual Student fit. Pretrained weights have separate training-data terms; see note below. |
| ResNet18 IMAGENET1K_V1 weights | resnet18-f37072fd.pth | TorchVision pretrained weights | Training-data/model-use terms require separate check | Downloaded into ignored cache; not committed. Used as frozen 512-dimensional encoder, not a trained-from-scratch claim. |
| OpenCV (opencv-python-headless) | 5.0.0.93 tested | pypi | Apache-2.0 | Frame extraction, evidence rendering. |
| FastAPI / Pydantic / SQLAlchemy / uvicorn / httpx | latest | pypi | MIT / MIT / MIT / BSD-3 / BSD-3 | Backend stack. |
| Next.js / React | 15.5.27 / 19.3.0 tested | npm | MIT | Frontend stack. |
| SmolVLM-500M-Instruct | a7da5b986cb59b408707209984f360a5f4ad7e47 | HuggingFaceTB model card | Apache-2.0 | Actually loaded on the local GPU. Candidate likelihood scoring is disclosed separately from failed generated-JSON labeling. Weights not committed. |
| Qwen2.5-VL-7B-Instruct | cc594898137f460bfe9f0759e9844b3ce807cfb5 planned | Qwen model card | Apache-2.0 | Private Vast setup was stopped and destroyed; remote weights/inference remain UNVERIFIED. |
| AutoLearn pilot drawings | synthetic-occupancy-v1 | Original project procedural generator | Original generator code: AGPL-3.0-only; raw pilot assets excluded from public release | 24 images, 12 rendering sessions; human confirmed the exact displayed labels. No customer footage, faces, plates or real-world quality claim. Raw assets remain ignored. |
| Demo video content (`data/demo/demo_bus.mp4`, historical M1 smoke) | generated | built by `scripts/make_demo_video.py` from the Ultralytics sample photo `bus.jpg` (bundled with the pip package) | package: AGPL-3.0; image subject rights: **unverified** | Translates the *whole* photo (bus **and** foreground pedestrians) across a canvas. Generation method declared per brief rules. Not committed to git. See the privacy note below — this clip is retained for the historical M1 smoke only. |
| Demo-safe video content (`data/demo/demo_bus_private_crop.mp4`) | generated | built by `scripts/make_demo_safe_video.py` from a **bus-only crop** `[350,220,755,665]` of the same `bus.jpg` | package: AGPL-3.0; image subject rights: **unverified** | The crop excludes the foreground pedestrians and still yields genuine YOLO `bus` detections. Synthetic left→right motion on a flat canvas; only the motion is synthetic, never the prediction. Used for the demo-safe journey. Not committed to git. |

## Task materials

The official Final Task is stored as text at `docs/source/README.md`
(provided by the hackathon organizers; redistribution limited to this repo).

## Upstream asset and license links

- Ultralytics repository and license (AGPL-3.0):
  <https://github.com/ultralytics/ultralytics> ·
  <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- Sample asset `bus.jpg` ships inside the `ultralytics` pip package
  (`ultralytics/assets/bus.jpg`). The **package** is AGPL-3.0; this attributes
  the upstream package/repository license only.
- [Ultralytics upstream licensing terms](https://www.ultralytics.com/legal/agpl-3-0-software-license).
- [TorchVision pretrained model terms](https://docs.pytorch.org/vision/stable/models.html)
  explicitly distinguish library licensing from terms derived from model
  training datasets. Commercial permission for the selected encoder weights
  is not established by this dependency inventory.
- Model cards: [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct),
  [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

## Privacy

The original `bus.jpg` sample contains **recognizable pedestrians**. The fact
that the image is publicly downloadable with the package does **not** establish
the subjects' consent, and we make **no such claim**: the subject/portrait
rights of the people in that image are **unverified**.

For this reason the demo-safe journey uses `demo_bus_private_crop.mp4`, a
**bus-only crop that excludes the foreground pedestrians**. The historical M1
smoke clip (`demo_bus.mp4`, whole photo) is retained unchanged for the M1 smoke
only and is not used in the demo-safe journey.

No private or customer footage is used. Faces/plates of non-consenting people
must not be uploaded during demos.

## Before release

- [x] Owner attested all copyright holders consented to AGPL-3.0-only for current original source.
- [ ] Publish and verify corresponding source for exact accepted deployed version.
- [x] Third-party libraries, models, and data recorded above.
- [ ] Final attribution pass before submission.

## Current natural automotive inputs (runtime only)

The VIDEO engineering inputs are genuine controlled car-following recordings
from Jiang R, Hu M, Zhang H, Gao Z, Jia B, Wu Q, Wang B, Yang M (2014),
*Traffic Experiment Reveals the Nature of Car-Following*, PLOS ONE 9(4): e94351,
[primary article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0094351).
Its ethics statement reports institutional review approval and written informed
consent from all drivers. These are road-traffic proxies, not service-bay footage
or independent quality benchmarks. Privacy inspection sampled actual frames;
it is not a forensic guarantee for all frames or a license over personal rights.

- [S1 on Commons](https://commons.wikimedia.org/wiki/File:Traffic-Experiment-Reveals-the-Nature-of-Car-Following-pone.0094351.s001.ogv):
  **CC BY 4.0**; original SHA256
  `a633b03fde02978cfc118d38fdc8a0f46398cde53800aa2ed41a54daf15cf26f`.
  Input uses source 0–15 s, static crop x=140,y=80,w=500,h=160,
  no audio, H.264 CRF18 at 25 fps; output SHA256
  `835b767f0ffa60467422f39d0436eb8f2aa7f00a97341f3eee30cf882ea9fd67`.
- [S2 on Commons](https://commons.wikimedia.org/wiki/File:Traffic-Experiment-Reveals-the-Nature-of-Car-Following-pone.0094351.s002.ogv):
  **CC BY 4.0**; a different original recording in the same study, original SHA256
  `6ccf4dbf0b4c8aae2529c78ce4654afe13ab2e5e4c5ec17e0c645efbbfbe56e8`.
  Input uses source 8–20 s, static crop x=0,y=140,w=450,h=150,
  no audio, H.264 CRF18 at 25 fps; output SHA256
  `027e048a23fd3d908cba4ddc24ec81ed320320cb6a94fdeddec211e129f5bf77`.

The [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) apply to these
recordings separately from the software license. Attribute the authors, article,
source and license, and disclose cropping/re-encoding if media are redistributed.
No raw recording, contact sheet, user upload or event evidence is included in the
source-only archive. Reproduction downloads the original primary-source files;
no synthetic motion, precomputed predictions or event annotations are inserted.

## Source offer and upstream separation

The [GNU AGPL v3 text](https://www.gnu.org/licenses/agpl-3.0.html) and
[GNU licensing guidance](https://www.gnu.org/licenses/gpl-howto.html) are the
basis of the included license and notices. The complete deployed corresponding
source must be freely downloadable through the prominent application source
link, including the original integration and build/install scripts. The selected
new source-release URL is documented in README and NOTICE; it is not live yet.

Ultralytics software/tracker and YOLO11n weights remain upstream AGPL-3.0.
Preserve upstream notices and the exact deployed weights' source/hash. The
source archive does not silently relicense weights or include model caches.
Make the unchanged pretrained weights available through their official source,
record the exact release URL and SHA256, and verify availability for the final
release; publish any modified covered weights and their necessary source if
used. There are no VisionFlow-trained YOLO weights in this release. See
[Ultralytics licensing](https://www.ultralytics.com/license) and
[upstream assets](https://github.com/ultralytics/assets).

SmolVLM and Qwen model-card licenses, the TorchVision weights' separate terms,
and historical sample-image subject-rights limitations above remain distinct.
Blocked AutoLearn/Teacher experiments are not accepted production capabilities.
The organizers' task text is excluded from the public source archive because its
redistribution permission is not established; it is not application source.
Publishing our source is not a guarantee about all external rights or security.

The current local VIDEO candidate and accepted fallback use an unchanged
`yolo11n.pt` file of 5,613,764 bytes, SHA256
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
This task verified the local bytes only; the final publisher must confirm the
exact upstream release/download URL and availability, and compare the deployed
bytes. Code is not a substitute for the selected weights and their provenance.
