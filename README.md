# Scratch Implementations

This repository contains from-scratch implementations of various Machine Learning models and concepts using PyTorch/Numpy/Python.

## Setup

### Environment

It is recommended to use a virtual environment to manage the dependencies. You can create a virtual environment using `venv` or `conda`.

**Using `venv`:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**Using `conda`:**

```bash
conda create -n torch-implementations python=3.12
conda activate torch-implementations
```

### Dependencies

Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Project Structure

The repository is organized into folders, each containing a specific model or concept implementation.

- **`BERT/`**: Implementation of the BERT model from scratch, including a custom tokenizer and training script.
- **`llm-inference/`**: Scripts for running inference with Large Language Models, including various sampling techniques.
- **`Machine-Learning/`**: Implementations of classic Machine Learning algorithms like K-Mode, MLP, and Perceptron.
- **`Modern-BERT/`**: An experimental implementation of a modern BERT architecture.
- **`n-grams/`**: Implementation of n-gram language models.
- **`optimizer-test/`**: A project to test and compare different optimization algorithms.
- **`quantization/`**: Notebook exploring model quantization techniques.
- **`smol-lm/`**: An attempt to build a small Language Model.
- **`Tokenizers/`**: Implementations of different tokenization algorithms like BPE, WordPiece, and an image tokenizer.
- **`transformers/`**: A notebook with various transformer-based model implementations.
- **`word2vec/`**: Notebooks for CBOW and Skip-gram implementations of word2vec.

## Usage

Each folder contains its own set of scripts and notebooks. Please refer to the individual files for specific instructions on how to run them.
