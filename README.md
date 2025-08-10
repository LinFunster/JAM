## JAM

This repository contains the official PyTorch implementation of our paper:

[IROS 2025] JAM: Keypoint-Guided Joint Prediction after Classification-Aware Marginal Proposal for Multi-Agent Interaction [[arxiv](https://arxiv.org/abs/2507.17152)]

### News
- **[10/08/2025]**: Our code is released!
- **[24/07/2025]**: We release the JAM paper on [arxiv](https://arxiv.org/abs/2507.17152)!
- **[01/07/2025]**: JAM is accepted by IROS 2025🎉!

## Data Preprocessing and Training
Download the Waymo Open Motion Dataset v1.2.0 from the [official website](https://waymo.com/open/download/). Use the data from `scenario/training` and `scenario/validation_interactive` for training and validation.

We use `Python 3.10` to ensure compatibility with the `waymo-open-dataset-tf-2-12-0` package.

```shell
# data preprocess
python data_process.py \
--load_path /JAM/waymo_dataset_1_2/training \ 
--save_path /JAM/waymo_dataset_1_2/processed_train \
--use_multiprocessing \
--processes=16

python data_process.py \
--load_path /JAM/waymo_dataset_1_2/validation_interactive \
--save_path /JAM/waymo_dataset_1_2/submission_validset \
--use_multiprocessing \
--processes=32 \
--test

# training & validation
bash train_JAM.sh

# viusalization
python visualization.py \
--name "jam_visual_res" \
--test_set "/Jam/waymo_dataset_1_2/validation_interactive" \
--jam_model_path "/Jam/jam_log/jam/epochs_29.pth"

# submission
python submission_interactive.py \
--name "jam" \
--sub_output_dir "/JAM/submission_pkg" \
--test_set "/JAM/waymo_dataset_1_2/submission_validset" \
--jam_model_path "/JAM/jam_log/jam/epochs_29.pth"
```

### Contact

If you have any questions or suggestions about this repo, please feel free to contact us (linfangze2023@email.szu.edu.cn).

### Citation

If you find JAM useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```BibTeX
@article{lin2025jam,
  title={JAM: Keypoint-Guided Joint Prediction after Classification-Aware Marginal Proposal for Multi-Agent Interaction},
  author={Lin, Fangze and He, Ying and Yu, Fei and Zhang, Hong},
  journal={arXiv preprint arXiv:2507.17152},
  year={2025}
}
```

### Acknowledgement
>We gratefully acknowledge the following projects for their inspiration: [GameFormer](https://github.com/MCZhi/GameFormer), [PP-TIL](https://github.com/LinFunster/PP-TIL), [BeTop](https://github.com/OpenDriveLab/BeTop/tree/main), [MTR](https://github.com/sshaoshuai/MTR), [QCNet](https://github.com/ZikangZhou/QCNet). Many thanks for their excellent contributions to the community.
