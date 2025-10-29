# Diabetes Prediction Modular Computational Pipeline (MCP) Toolset

This repository contains the Modular Computational Pipeline (MCP) Toolset for predicting diabetes using the PIMA Indians Diabetes Database. This toolset provides a structured and flexible way to interact with the entire machine learning workflow, from data exploration and preparation to model training, inference, and evaluation.

## Features

The MCP Toolset exposes the following key capabilities via a REST API:

*   **Load Data:** Import and preprocess datasets in preparation for analysis or model inference.
*   **Run Model Inference:** Execute trained machine learning models on input data to generate predictions.
*   **Show Results:** Display results in a clear and interpretable format, including tables, plots, and summaries.
*   **Evaluate Model:** Assess the performance of the models using appropriate evaluation metrics, enabling users to understand model accuracy and reliability.

This modular approach promotes reproducibility, flexibility, and ease of use for diabetes prediction tasks.

## Project Structure

```
mcp_toolset/
├── app.py                  # Flask application for the REST API
├── server.py               # Core logic for data processing, model training, and evaluation
├── requirements.txt        # Python dependencies
├── data/
│   └── diabetes.csv        # PIMA Indians Diabetes Database
├── trained_models/         # Directory for trained machine learning models
│   ├── constant_model.pkl
│   ├── logistic_regression.pkl
│   ├── neural_network.pth
│   ├── random_forest.pkl
│   └── xgboost.pkl
└── results/                # Directory for generated plots and processed data
    ├── class_distribution.png
    ├── cross_validation_accuracies.png
    ├── feature_distributions.png
    ├── missing_data_heatmap.png
    ├── split_class_distributions.png
    ├── X_test.csv
    ├── X_train.csv
    ├── y_test.csv
    ├── y_train.csv
    ├── roc_curves_all_models.png
    ├── confusion_matrices.png
    └── feature_importance_subplots.png
```

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd mcp_toolset
    ```
    (Note: Replace `<repository_url>` with the actual URL of your GitHub repository once it's created.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the MCP Server

To start the Flask API server, navigate to the `mcp_toolset` directory and run:

```bash
python app.py
```

The server will typically run on `http://127.0.0.1:5000` (or another port if configured).

### API Endpoints

The server exposes several endpoints corresponding to the MCP tools. You can interact with them using `curl`, Postman, or any HTTP client.

**1. Data Exploration**
*   **Endpoint:** `/run_data_exploration`
*   **Method:** `GET`
*   **Description:** Performs initial data exploration and generates visualizations.
*   **Example:**
    ```bash
    curl http://127.0.0.1:5000/run_data_exploration
    ```

**2. Prepare Data**
*   **Endpoint:** `/prepare_data`
*   **Method:** `POST`
*   **Description:** Loads, cleans, scales, and splits the dataset into training, validation, and test sets.
*   **Parameters (Optional in JSON body):**
    *   `test_size`: (float, default: 0.2) Proportion of the dataset to include in the test split.
    *   `random_state`: (int, default: 42) Controls the shuffling applied to the data before applying the split.
*   **Example:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"test_size": 0.25, "random_state": 123}' http://127.0.0.1:5000/prepare_data
    ```
    or for default parameters:
    ```bash
    curl -X POST http://127.0.0.1:5000/prepare_data
    ```

**3. Train Model**
*   **Endpoint:** `/train_model`
*   **Method:** `POST`
*   **Description:** Trains a specified machine learning model and saves it.
*   **Parameters (Required in JSON body):**
    *   `model_name`: (string) The name of the model to train.
        *   Accepted values: `"constant_model"`, `"logistic_regression"`, `"random_forest"`, `"xgboost"`, `"neural_network"`
*   **Example:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"model_name": "random_forest"}' http://127.0.0.1:5000/train_model
    ```

**4. Get Cross-Validation Accuracies**
*   **Endpoint:** `/get_cv_accuracies`
*   **Method:** `GET`
*   **Description:** Performs 5-fold cross-validation for all implemented models and returns their mean accuracy and standard deviation.
*   **Example:**
    ```bash
    curl http://127.0.0.1:5000/get_cv_accuracies
    ```

**5. Generate ROC Curves**
*   **Endpoint:** `/generate_roc_curves`
*   **Method:** `GET`
*   **Description:** Generates a single figure with ROC curve subplots for all models.
*   **Example:**
    ```bash
    curl http://127.0.0.1:5000/generate_roc_curves
    ```

**6. Generate Confusion Matrices**
*   **Endpoint:** `/generate_confusion_matrices`
*   **Method:** `GET`
*   **Description:** Generates a figure with confusion matrices for all models.
*   **Example:**
    ```bash
    curl http://127.0.0.1:5000/generate_confusion_matrices
    ```

**7. Generate Feature Importance**
*   **Endpoint:** `/generate_feature_importance`
*   **Method:** `GET`
*   **Description:** Generates a 2x2 subplot figure comparing feature importances for the main models.
*   **Example:**
    ```bash
    curl http://127.0.0.1:5000/generate_feature_importance
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
