# NNgTL

## Overview

This project implements **NNgTL**, as described in the paper "[NNgTL: Neural Network Guided Optimal Temporal Logic Task Planning for Mobile Robots](https://ieeexplore.ieee.org/abstract/document/10611699/)".

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/LRJ-2000/NNgTL.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd NNgTL
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Pretrained Models

Pretrained models trained on randomly generated LTL formulae and workspaces are available. These models can be tested in the provided scenarios or used for transfer learning on custom LTL task templates and workspace scenarios.

Download the [pretrained models](https://drive.google.com/drive/folders/1VnAZl7_gLNtNS9BJCBU8DLq3_Cvu5dnF?usp=sharing) and place them in `./model/pretrained_model/`.

The repository structure should be:

```
NNgTL/
├── model/
│   └── pretrained_model/
│       ├── BA_Predict/BA_Predict_pretrained.pth   # StateNet
│       └── LTL_Net/LTL_Net_pretrained.pth         # PathNet
•   •   •
•   •   •
```

## LTL Formulae

[Spot](https://spot.lre.epita.fr/) is used to generate random LTL formulae, saved in `./utils/LTL.txt`. These can be used to test the algorithms or replaced with custom LTL formulae.

## Usage

### Generate Dataset

To generate the training dataset:

```bash
python ./scripts/generate_dataset.py --num_samples 5000
```

The `--num_samples` parameter specifies the number of samples to generate.

### Train the Model

Train the neural network models:

```bash
python ./scripts/training.py -m BA_Predict
python ./scripts/training.py -m LTL_Net
```

**Script Parameters:**

- `-m`, `--model`: Model to train (`BA_Predict` or `LTL_Net`). Corresponds to "StateNet" and "PathNet" in the paper. Default: `BA_Predict`.
- `-s`, `--scale`: Model scale (`tiny`, `small`, `medium`, `large`). Use `large` to train from the pretrained model. Default: `large`.
- `-t`, `--type`: Training type (`1` for scratch, `2` for checkpoint continuation, `3` for transfer learning). Default: `1`.
- `-c`, `--checkpoint`: Path to the checkpoint file. Default: `./model/checkpoint/`.
- `-p`, `--pretrained_model`: Path to the pretrained model. Default: `./model/pretrained_model/`.
- `-b`, `--batch_size`: Training batch size. Default: `128`.
- `-e`, `--epochs`: Number of training epochs. Default: `200`.
- `-i`, `--save_interval`: Save checkpoint every `n` epochs. Default: `10`.

### Run the Planning Algorithms

*Note: To use the pretrained model without training, add `--use_pretrained_model` to each command.*

- **Test on a given LTL formula:**

  ```bash
  python main.py --LTL "[]<> e1 && (NOT e1 U e2) && <> e3"
  ```

- **Generate random testing data:**

  ```bash
  python main.py --generate_data 100
  ```

- **Test on specific testing data:**

  ```bash
  python main.py --data_id 1
  ```

- **Test on all testing data:**

  ```bash
  python main.py --test_algorithm
  ```

**Script Parameters:**

- `--LTL`: LTL formula to test.
- `--generate_data`: Generate testing data with specified number of samples.
- `--data_id`: ID of the data sample to test.
- `--model_scale`: Network model scale. Default: `large`.
- `--visualize_path`: Visualize the path.
- `--use_pretrained_model`: Use pretrained model.
- `--save_data`: Save data.
- `--test_algorithm`: Test the algorithm.
- `--test_unbiased`: Test with unbiased sampling strategy.

**Parameters for Tree Construction:**

- `--n_max`: Maximum number of iterations. Default: `4000`.
- `--max_time`: Maximum allowed time for tree construction. Default: `200`.
- `--size_N`: Size of the discrete workspace. Default: `200`.
- `--is_lite`: Use lite version (excluding extending and rewiring).
- `--weight`: Weight parameter. Default: `0.2`.
- `--p_closest`: Probability of choosing node `q_p_closest`. Default: `0.9`.
- `--y_rand`: Probability when deciding the target point. Default: `0.8`.
- `--step_size`: Step size used in the `near` function. Default: `0.8`.
- `--p_BA_predict`: Probability of using BA_Predict's prediction. Default: `0.8`.

## Project Structure

- `utils/`: Utility modules for tasks, Büchi automaton parsing, workspace handling, datasets, and models.
- `scripts/training.py`: Script for training neural network models.
- `scripts/generate_dataset.py`: Script for generating the dataset.
- `main.py`: Main script for running planning algorithms and testing models.

## Acknowledgements

This implementation is based on [TLRRT_star](https://github.com/XushengLuo92/TLRRT_star).

Utilizes [Spot](https://spot.lre.epita.fr/) for generating random LTL formulae and [LTL2BA](http://www.lsv.fr/~gastin/ltl2ba/) for converting LTL formulae into Büchi automata.

## Citation

If you find this project useful, please cite:

```
@inproceedings{liu2024nngtl,
  title={NNgTL: Neural Network Guided Optimal Temporal Logic Task Planning for Mobile Robots},
  author={Liu, Ruijia and Li, Shaoyuan and Yin, Xiang},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={10496--10502},
  year={2024},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License.