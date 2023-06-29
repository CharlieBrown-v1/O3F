# Object-Oriented Option Framework for Robotics Manipulation in Clutter

The Official Code for ["Object-Oriented Option Framework for Robotics Manipulation in Clutter"]

## Code Structure
O3F

    |- gym: modified gym based on our needs

    |- models: directory for saving by-product models
    
    |- stable_baselines3: modified stable-baselines3 (sb3) based on our needs

    |- train_oe.py: code for Option Executor (OE) training

    |- train_op.py: code for Option Planner (OP) training

## Setup
1. Please finish the following steps to install conda environment and related python packages
    - Conda Environment create
    ```bash
    conda create --name <env_name> --file spec-list.txt
    ```
    - Package install
    ```bash
    pip install -r requirements.txt
    ```
2. The environments used in this work require MuJoCo as dependency. Please setup it following the instructions:
    - Instructions for MuJoCo: https://mujoco.org/

## Using

We offer a fine-trained Option Executor (OE) model which Option Planner (OP) utilizes during training for testing.

```bash
models/execute.zip
```

* Training of OE
    * Command
    ```bash
    python train_oe.py
    ```
    * The models of oe will be saved at `oe_model`.
    * The tensorboard logs of oe will be saved at `oe_train`.

* Training of OP
    * Command
    ```bash
    python train_op.py
    ```
    * The models of op will be saved at `op_model`.
    * The tensorboard logs of op will be saved at `op_train`.
