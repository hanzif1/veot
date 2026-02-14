---
license: bsd-3-clause
---

# VideoMind Datasets

<div style="display: flex; gap: 5px;">
  <a href="https://arxiv.org/abs/2503.13444" target="_blank"><img src="https://img.shields.io/badge/arXiv-2503.13444-red"></a>
  <a href="https://videomind.github.io/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-brightgreen"></a>
  <a href="https://github.com/yeliudev/VideoMind/blob/main/README.md" target="_blank"><img src="https://img.shields.io/badge/License-BSD--3--Clause-purple"></a>
  <a href="https://github.com/yeliudev/VideoMind" target="_blank"><img src="https://img.shields.io/github/stars/yeliudev/VideoMind"></a>
</div>

This repository provides the videos and annotations of **VideoMind-SFT** and downstream evaluation benchmarks. All the videos are provided in both **original files** and **compressed versions (3 FPS, 480p, no audio)**. A complete list of the datasets is as follows. Please download the sub-directories accordingly if you only need part of the data.

### VideoMind-SFT (481K)

#### Grounder (210K):

| Dataset | Directory | Source Link |
|-|-|-|
| QVHighlights | [`qvhighlights`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/qvhighlights) | https://github.com/jayleicn/moment_detr |
| DiDeMo | [`didemo`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/didemo) | https://github.com/LisaAnne/LocalizingMoments/ |
| TACoS | [`tacos`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/tacos) | https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus |
| QuerYD | [`queryd`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/queryd) | https://www.robots.ox.ac.uk/~vgg/data/queryd/ |
| HiREST (Grounding) | [`hirest`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/hirest) | https://github.com/j-min/HiREST |
| HiREST (Step Captioning) | [`hirest`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/hirest) | https://github.com/j-min/HiREST |
| CosMo-Cap | [`cosmo_cap`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/cosmo_cap) | https://github.com/showlab/cosmo |
| InternVid-VTime | [`internvid_vtime`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/internvid_vtime) | https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid |

#### Verifier (232K):

| Dataset | Directory | Source Link |
|-|-|-|
| QVHighlights-Verify | [`verifying`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/verifying), [`qvhighlights`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/qvhighlights) | https://github.com/jayleicn/moment_detr |
| DiDeMo-Verify | [`verifying`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/verifying), [`didemo`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/didemo) | https://github.com/LisaAnne/LocalizingMoments/ |
| TACoS-Verify | [`verifying`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/verifying),[`tacos`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/tacos) | https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus |

#### Planner (39K):

| Dataset | Directory | Source Link |
|-|-|-|
| NExT-QA-Plan | [`planning`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/planning), [`nextqa`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/nextqa) | https://github.com/doc-doc/NExT-QA |
| QVHighlights-Plan | [`planning`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/planning), [`qvhighlights`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/qvhighlights) | https://github.com/jayleicn/moment_detr |

### Benchmarks

| Dataset | Type | Directory | Source Link |
|-|:-:|-|-|
| CG-Bench | Grounded VideoQA | [`cgbench`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/cgbench) | https://huggingface.co/datasets/CG-Bench/CG-Bench |
| ReXTime | Grounded VideoQA | [`rextime`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/rextime), [`activitynet`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/activitynet), [`qvhighlights`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/qvhighlights) | https://github.com/ReXTime/ReXTime |
| NExT-GQA | Grounded VideoQA | [`nextgqa`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/nextgqa) | https://github.com/doc-doc/NExT-GQA |
| Charades-STA | VTG | [`charades_sta`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/charades_sta) | https://github.com/jiyanggao/TALL |
| ActivityNet-Captions | VTG | [`activitynet_captions`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/activitynet_captions), [`activitynet`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/activitynet) | https://cs.stanford.edu/people/ranjaykrishna/densevid/ |
| QVHighlights | VTG | [`qvhighlights`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/qvhighlights) | https://github.com/jayleicn/moment_detr |
| TACoS | VTG | [`tacos`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/tacos) | https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus |
| Ego4D-NLQ | VTG | [`ego4d_nlq`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/ego4d_nlq), [`ego4d`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/ego4d) | https://ego4d-data.org/ |
| ActivityNet-RTL | VTG | [`activitynet_rtl`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/activitynet_rtl), [`activitynet`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/activitynet) | https://github.com/NVlabs/LITA |
| Video-MME | General VideoQA | [`videomme`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/videomme) | https://github.com/BradyFU/Video-MME |
| MLVU | General VideoQA | [`mlvu`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/mlvu) | https://github.com/JUNJIE99/MLVU |
| LVBench | General VideoQA | [`lvbench`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/lvbench) | https://github.com/THUDM/LVBench |
| MVBench | General VideoQA | [`mvbench`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/mvbench) | https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md |
| LongVideoBench | General VideoQA | [`longvideobench`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/longvideobench) | https://github.com/longvideobench/LongVideoBench |

The following datasets are not used in our project (partially used during early exploration), but we still share them to facilitate future research.

| Dataset | Type | Training | Evaluation | Directory | Source Link |
|-|:-:|:-:|:-:|-|-|
| QaEgo4D | Grounded VideoQA | ✅ | ✅ | [`qa_ego4d`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/qa_ego4d), [`ego4d`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/ego4d) | https://github.com/lbaermann/qaego4d |
| Ego4D-NaQ | VTG | ✅ | ✅ | [`ego4d_naq`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/ego4d_naq), [`ego4d`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/ego4d) | https://github.com/srama2512/NaQ |
| Ego-TimeQA | VTG | ✅ | ❌ | [`ego_timeqa`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/ego_timeqa), [`ego4d`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/ego4d) | https://github.com/Becomebright/GroundVQA |
| Vid-Morp | VTG | ✅ | ❌ | [`vid_morp`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/vid_morp) | https://github.com/baopj/Vid-Morp |
| VideoXum | VTG (originally VS) | ✅ | ✅ | [`videoxum`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/videoxum) | https://github.com/jylins/videoxum |
| YouCook2 | VTG (originally DVC) | ✅ | ✅ | [`youcook2`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/youcook2) | http://youcook2.eecs.umich.edu/ |
| STAR | VideoQA | ✅ | ✅ | [`star`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/star), [`charades_sta`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/charades_sta) | https://bobbywu.com/STAR/ |
| COIN | - | - | - | [`coin`](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/coin) | https://github.com/coin-dataset/annotations |

**Notes**:

1. For some datasets (e.g., ReXTime), the annotations and videos are stored in different folders. All the directories in `Directory` need to be downloaded.
2. Use the following commands to concatenate and extract video tar splits (e.g., videos.tar.gz.00, videos_3fps_480_noaudio.tar.gz.00).

```
# videos.tar.gz.00, videos.tar.gz.01
cat videos.tar.gz.* | tar -zxvf -

# videos_3fps_480_noaudio.tar.gz.00, videos_3fps_480_noaudio.tar.gz.01
cat videos_3fps_480_noaudio.tar.gz.* | tar -zxvf -
```

## 📖 Citation

Please kindly cite our paper if you find this project helpful.

```
@article{liu2025videomind,
  title={VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning},
  author={Liu, Ye and Lin, Kevin Qinghong and Chen, Chang Wen and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2503.13444},
  year={2025}
}
```
