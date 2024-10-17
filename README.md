# Poisonous Mushroom Predictor

## Project Overview

This project implements a decision tree classifier from scratch to predict whether a mushroom is poisonous or edible. The dataset used for this classification task contains various features related to mushroom characteristics (e.g., cap shape, color, gill attachment). The decision tree supports three different splitting criteria:

- Gini impurity
- Scaled entropy
- Squared impurity

It uses **grid search** to find the best hyperparameters (e.g., `max_depth` and `split_function`), and it evaluates the performance using **zero-one loss**.

Additionally, this project includes a Jupyter notebook for interactive exploration and testing.

## Requirements

To run this project, the following dependencies are needed. Install them using the `requirements.txt` file:

```bash
pandas
numpy
scikit-learn
joblib
```

## Data Preprocessing
### Data Cleaning 
Missing values in the dataset are handled by:

Dropping columns with more than 80% missing values.
Imputing missing values in other columns with the most frequent value.
### Feature Encoding

One-Hot Encoding is applied to nominal categorical features.
Label Encoding is applied to ordinal categorical features.

## Decision Tree Model
The DecisionTree class provides the core functionality for training and predicting with a decision tree. The decision tree is grown recursively, and leaf nodes are determined based on criteria such as max_depth, min_samples_split, and the chosen split function (gini, scaled_entropy, or squared_impurity).

### Parameters
- max_depth: The maximum depth the tree can reach before stopping.
- min_samples_split: The minimum number of samples required to split a node.
- split_function: Determines the criterion used to split the nodes (e.g., gini, scaled_entropy, or squared_impurity).

### Grid Search
A grid search function is provided to automatically evaluate combinations of hyperparameters (such as different max_depth values and split functions) using cross-validation. The best hyperparameters are selected based on minimizing the zero-one loss.

## Running the Project
### Data Loading
The dataset (secondary_data.csv) should be placed in the data/ folder. The script automatically loads this dataset.

### Interactive Execution
The program provides an interactive prompt allowing you to:

- Run grid search to find the best parameters.
- Use custom parameters for the decision tree.
- Evaluation: After training, the model's performance on the test set is evaluated using zero-one loss. Depending on the performance, feedback is provided to guide improvements.

## How to Use
- Clone the repository:

````bash
git clone https://github.com/hmente/PoisonousMushroomPredictor.git
````
Install the dependencies:

````bash
pip install -r requirements.txt
````
Run the script:

````bash
python main.py
````
Use the interactive prompts to either run grid search or provide custom parameters for training.

(Optional) You can explore the Jupyter notebook for interactive analysis.

## Example Usage
After running the script, you will be prompted to choose between using grid search to find the best parameters or providing custom parameters manually. For example:

- Grid Search: The script will test different combinations of max_depth and split_function and report the best performing model based on zero-one loss.

- Custom Parameters: You can input custom values for max_depth and split_function, and the decision tree will be trained with these parameters.

## Results & Evaluation
After training, the model is evaluated using zero-one loss, which measures the proportion of incorrect predictions. Based on the test set loss, feedback will be provided on the model's performance.

## Additional Information
- Interactive Flow: The script allows the user to choose between grid search or providing custom parameters.
- Jupyter Notebook: An additional Jupyter notebook is provided in the codebase for interactive exploration and testing of the decision tree classifier.
