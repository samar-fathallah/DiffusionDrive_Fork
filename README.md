# DiffusionDrive – NeuroNCAP Integration Fork

This repository is a fork of the original  
[DiffusionDrive](https://github.com/OpenDriveLab/DiffusionDrive)  
and adapts it for integration with **NeuroNCAP** (Neuro-symbolic Collaborative Assessment Platform), a standardized framework for evaluating autonomous-driving (AD) models.

The purpose of this fork is to demonstrate how to implement a trajectory-prediction model so it can be evaluated using the **NeuroNCAP evaluation framework**.

---

## 🔧 Changes in This Fork

This repository differs from the original DiffusionDrive repository in the following ways:

### 1. **Inference Configuration**
A custom configuration file was added:

- `inference/inference_e2e.py`

This config limits the operations applied to the input during inference, ensuring compatibility with the NeuroNCAP evaluation pipeline.

---

### 2. **Complete Inference Functionality**

The `inference/` directory contains two new components to enable standalone inference:

#### `runner.py`
A wrapper around the original DiffusionDrive model that enables running inference independently of training/validation.  
The original repository is designed primarily around the training loop, so this wrapper:

- loads and initializes the model
- prepares the input in a lightweight manner
- performs forward inference
- formats outputs according to NeuroNCAP requirements

#### `server.py`
A simple FastAPI server that exposes inference endpoints.  
These endpoints follow the **NeuroNCAP Model API Specification**, allowing the NeuroNCAP system to send inputs and retrieve predictions.

This turns DiffusionDrive into a standalone, API-driven model node compatible with NeuroNCAP experiments.

---

### 3. **Docker/Singularity Support**

A `Dockerfile` was added to build an image for deployment.  
This Docker image was used to generate a `.sif` (Singularity image) required by the NeuroNCAP model execution environment.

---

## 📁 Repository Structure (Additions)

