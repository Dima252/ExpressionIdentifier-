# Expression Identifier

This project is a binary image classification system designed to recognize and distinguish between "Happy" and "Sad" facial expressions.

## Project Structure

*   **`ExpressionIdentifier.ipynb`**: A Jupyter Notebook exploring expression identification. This typically contains initial data exploration, model training using higher-level wrappers or pre-trained models (like FastAI), and performance evaluation.
*   **`expressionidentifier_exercise_4.ipynb`**: A Jupyter Notebook demonstrating how to build and train an image classification model **from scratch using PyTorch**. It includes explicit definitions for data transformations (resizing to 64x64, normalization), dataset loading using `torchvision.datasets.ImageFolder`, a training/validation split (80%/20%), and custom DataLoader configurations.
*   **`data/`**: The directory containing the dataset images, organized by class.
    *   `data/Happy/`: Contains images of happy facial expressions.
    *   `data/Sad/`: Contains images of sad facial expressions.

## Requirements

To run the notebooks, you will need the following libraries:

*   `torch` (PyTorch)
*   `torchvision`
*   `matplotlib`
*   `numpy`
*   `jupyter`

## Usage

1.  Make sure you have your dataset arranged correctly in the `data/` folder, with `Happy` and `Sad` subdirectories containing the respective images.
2.  Open the Jupyter Notebooks:
    ```bash
    jupyter notebook
    ```
3.  Run the cells in `ExpressionIdentifier.ipynb` or `expressionidentifier_exercise_4.ipynb` to train and evaluate the expression identification models.

## Authors
* Shalev Yosefashvili
* Dimitry Todoseyev
