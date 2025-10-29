"""
Main server file for the Diabetes Prediction MCP Toolset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Classes ---

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout_prob=0.2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(hidden_dim3, 1)
    def forward(self, x):
        out = self.dropout1(self.relu1(self.fc1(x)))
        out = self.dropout2(self.relu2(self.fc2(out)))
        out = self.dropout3(self.relu3(self.fc3(out)))
        out = self.fc4(out)
        return torch.sigmoid(out)

# --- Tool Implementations ---

def run_data_exploration() -> str:
    """
    Performs data exploration and generates plots.
    """
    print("Running data exploration...")
    # --- FIX: make path relative to this script file ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of this file
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')  # full path to CSV

    # Load CSV using the corrected path
    df = pd.read_csv(DATA_PATH)
    PLOTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df_original = pd.read_csv(DATA_PATH)
    df_nan = df_original.copy()

    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_nan[zero_features] = df_nan[zero_features].replace(0, np.nan)

    df_imputed = df_nan.copy()
    for col in zero_features:
        median = df_imputed[col].median()
        df_imputed[col] = df_imputed[col].fillna(median)

    df = df_imputed.copy()

    # Missing data heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_nan.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Data Heatmap')
    plt.savefig(os.path.join(PLOTS_DIR, 'missing_data_heatmap.png'))
    plt.close()

    # Feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(df.columns):
        ax = axes[i//3, i%3]
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_distributions.png'))
    plt.close()

    # Class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Outcome', data=df)
    plt.title('Class Distribution (0: No Diabetes, 1: Diabetes)')
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'))
    plt.close()

    # Split data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Split class distributions
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.countplot(x=y_train)
    plt.title('Train Set Class Distribution')
    plt.subplot(1, 3, 2)
    sns.countplot(x=y_val)
    plt.title('Validation Set Class Distribution')
    plt.subplot(1, 3, 3)
    sns.countplot(x=y_test)
    plt.title('Test Set Class Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'split_class_distributions.png'))
    plt.close()

    msg = f"Data exploration complete. Plots saved in '{PLOTS_DIR}'"
    print(msg)
    return msg

def prepare_data(test_size: float = 0.2, random_state: int = 42) -> str:
    """
    Loads, cleans, scales, and splits the diabetes dataset.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
    RESULTS_DIR = 'mcp_toolset/results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    missing_data_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[missing_data_features] = df[missing_data_features].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df[missing_data_features] = imputer.fit_transform(df[missing_data_features])
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train.to_csv(os.path.join(RESULTS_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(RESULTS_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(RESULTS_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(RESULTS_DIR, 'y_test.csv'), index=False)
    msg = f"Data prepared and saved successfully in '{RESULTS_DIR}'"
    print(msg)
    return msg

def train_model(model_name: str) -> str:
    """
    Trains a specified model and saves it to disk.
    """
    print(f"--- Training {model_name} ---")
    RESULTS_DIR = 'mcp_toolset/results'
    MODELS_DIR = 'mcp_toolset/trained_models'
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train = pd.read_csv(os.path.join(RESULTS_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(RESULTS_DIR, 'y_train.csv')).squeeze()
    save_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')

    if model_name == 'constant_model':
        model = DummyClassifier(strategy='most_frequent')
        model.fit(X_train, y_train)
        joblib.dump(model, save_path)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, save_path)
    elif model_name == 'random_forest':
        param_grid_rf = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [10, 20, 30, 40, 50, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']}
        rf_model_to_tune = RandomForestClassifier(random_state=42)
        rf_random_search = RandomizedSearchCV(estimator=rf_model_to_tune, param_distributions=param_grid_rf, n_iter=100, cv=5, verbose=0, random_state=42, n_jobs=-1)
        rf_random_search.fit(X_train, y_train)
        model = rf_random_search.best_estimator_
        joblib.dump(model, save_path)
    elif model_name == 'xgboost':
        param_grid_xgb = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [3, 4, 5, 6, 7, 8], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'subsample': [0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.7, 0.8, 0.9, 1.0]}
        xgb_model_to_tune = XGBClassifier(eval_metric='logloss', random_state=42)
        xgb_random_search = RandomizedSearchCV(estimator=xgb_model_to_tune, param_distributions=param_grid_xgb, n_iter=100, cv=5, verbose=0, random_state=42, n_jobs=-1)
        xgb_random_search.fit(X_train, y_train)
        model = xgb_random_search.best_estimator_
        joblib.dump(model, save_path)
    elif model_name == 'neural_network':
        save_path = os.path.join(MODELS_DIR, f'{model_name}.pth')
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        model = SimpleNN(X_train.shape[1])
        loss_function_nn = nn.BCELoss()
        optimizer_nn = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.02)
        num_epochs_nn = 10000
        for epoch in range(num_epochs_nn):
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                loss = loss_function_nn(outputs, y_batch)
                optimizer_nn.zero_grad()
                loss.backward()
                optimizer_nn.step()
        torch.save(model.state_dict(), save_path)
    else:
        raise ValueError("Unknown model_name.")

    print(f"Model trained and saved to {save_path}")
    return save_path

def get_cv_accuracies() -> dict:
    """
    Performs 5-fold cross-validation for all models and returns mean accuracy and standard deviation.
    """
    RESULTS_DIR = 'mcp_toolset/results'
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'diabetes.csv')
    df = pd.read_csv(DATA_PATH)
    missing_data_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[missing_data_features] = df[missing_data_features].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df[missing_data_features] = imputer.fit_transform(df[missing_data_features])
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'constant_model': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', max_depth=40, random_state=42),
        'xgboost': XGBClassifier(subsample=0.9, n_estimators=100, max_depth=3, learning_rate=0.05, colsample_bytree=0.7, eval_metric='logloss', random_state=42)
    }
    results = {}

    for name, model in models.items():
        print(f"  - Running CV for {name}...")
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        results[name] = {'mean_accuracy': np.mean(cv_scores), 'std_dev': np.std(cv_scores)}

    print("  - Running CV for neural_network...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    nn_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"    - Fold {fold+1}/5")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        model_nn = SimpleNN(X_train.shape[1])
        loss_fn = nn.BCELoss()
        optimizer = optim.SGD(model_nn.parameters(), lr=0.01, weight_decay=0.02)
        for _ in range(10000):
            for X_batch, y_batch in train_loader:
                outputs = model_nn(X_batch)
                loss = loss_fn(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model_nn.eval()
        with torch.no_grad():
            val_preds = (model_nn(X_val_tensor) >= 0.5).float()
            acc = accuracy_score(y_val_tensor, val_preds)
            nn_scores.append(acc)
    results['neural_network'] = {'mean_accuracy': np.mean(nn_scores), 'std_dev': np.std(nn_scores)}

    print("Cross-validation complete.")
    # Create a DataFrame from the results
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'Model'
    results_df.columns = ['Mean Accuracy', 'Standard Deviation']

    # Define the desired order of models
    model_order = ['constant_model', 'logistic_regression', 'random_forest', 'xgboost', 'neural_network']
    results_df = results_df.reindex(model_order)

    # Plotting the table
    # Create small figure — it will expand a little as needed by the table
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(results_df.index) + 0.5))
    ax.axis('off')

    # --- Style settings ---
    table = ax.table(
        cellText=results_df.round(3).values,
        colLabels=results_df.columns,
        rowLabels=results_df.index,
        loc='center',
        cellLoc='center'
    )

    # --- Modern design tweaks ---
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.3)  # adjust scaling for compactness

    # Header + cell styling
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.8)
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='#222222')
            cell.set_facecolor('#E8EEF1')
        else:
            cell.set_facecolor('#FAFAFA' if row % 2 == 0 else '#F2F2F2')

    # Tight bounding box so figure wraps around table perfectly
    fig.tight_layout(pad=0.05)

    # Save
    save_path = os.path.join(RESULTS_DIR, "cross_validation_accuracies.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=300, transparent=True)
    plt.close()

    print(f"✅ Cross-validation accuracies table saved to {save_path}")
    
    return results

def generate_all_roc_curves_plot() -> str:
    """
    Generates a single figure with ROC curve subplots for all models.
    """
    print("Generating ROC curve comparison plot for all models...")
    RESULTS_DIR = 'mcp_toolset/results'
    MODELS_DIR = 'mcp_toolset/trained_models'
    model_names = ['constant_model', 'logistic_regression', 'random_forest', 'xgboost', 'neural_network']
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    X_test = pd.read_csv(os.path.join(RESULTS_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(RESULTS_DIR, 'y_test.csv')).squeeze()

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl' if model_name != 'neural_network' else f'{model_name}.pth')
        if not os.path.exists(model_path):
            train_model(model_name)
        
        if model_path.endswith('.pkl'):
            model = joblib.load(model_path)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        elif model_path.endswith('.pth'):
            model = SimpleNN(X_test.shape[1])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            with torch.no_grad():
                y_pred_proba = model(torch.tensor(X_test.values, dtype=torch.float32)).numpy()

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='mediumseagreen', lw=4, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='orange', lw=4, linestyle='-', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=18)
        ax.set_ylabel('True Positive Rate', fontsize=18)
        ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=20)
        ax.legend(loc="lower right", fontsize=14)

    plt.tight_layout(pad=3.0)
    save_path = os.path.join(RESULTS_DIR, 'roc_curves_all_models.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve subplot figure saved to {save_path}")
    return save_path

def generate_feature_importance_subplots() -> str:
    """
    Generates a 2x2 subplot figure comparing feature importances for the main models.
    """
    print("Generating feature importance comparison plot...")
    RESULTS_DIR = 'mcp_toolset/results'
    MODELS_DIR = 'mcp_toolset/trained_models'
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'neural_network']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    X_test = pd.read_csv(os.path.join(RESULTS_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(RESULTS_DIR, 'y_test.csv')).squeeze()
    name_mapping = {'DiabetesPedigreeFunction': 'DiabetesPedigree\nFunction', 'BloodPressure': 'Blood\nPressure'}

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl' if model_name != 'neural_network' else f'{model_name}.pth')
        if not os.path.exists(model_path):
            train_model(model_name)

        importances = None
        if model_name == 'logistic_regression':
            model = joblib.load(model_path)
            importances = model.coef_[0]
        elif model_name in ['random_forest', 'xgboost']:
            model = joblib.load(model_path)
            importances = model.feature_importances_
        elif model_name == 'neural_network':
            model = SimpleNN(X_test.shape[1])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)
            with torch.no_grad():
                baseline_accuracy = accuracy_score((model(X_test_tensor) >= 0.5).float(), y_test_tensor)
                nn_importances = []
                for j in range(X_test.shape[1]):
                    X_test_permuted = X_test.copy().values
                    np.random.shuffle(X_test_permuted[:, j])
                    permuted_tensor = torch.tensor(X_test_permuted, dtype=torch.float32)
                    permuted_accuracy = accuracy_score((model(permuted_tensor) >= 0.5).float(), y_test_tensor)
                    nn_importances.append(baseline_accuracy - permuted_accuracy)
                importances = np.array(nn_importances)

        feature_importance = pd.Series(importances, index=X_test.columns).abs().sort_values(ascending=True)
        feature_importance.index = feature_importance.index.map(lambda x: name_mapping.get(x, x))
        ax.barh(feature_importance.index, feature_importance.values)
        ax.set_title(model_name.replace('_', ' ').title(), fontsize=16)
        ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout(pad=3.0)
    save_path = os.path.join(RESULTS_DIR, 'feature_importance_subplots.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")
    return save_path

def generate_confusion_matrices() -> str:
    """
    Generates a figure with confusion matrices for all models.
    """
    print("Generating confusion matrices for all models...")
    RESULTS_DIR = 'mcp_toolset/results'
    MODELS_DIR = 'mcp_toolset/trained_models'
    PLOTS_DIR = RESULTS_DIR
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = ['constant_model', 'logistic_regression', 'random_forest', 'xgboost', 'neural_network']
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    X_test = pd.read_csv(os.path.join(RESULTS_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(RESULTS_DIR, 'y_test.csv')).squeeze()

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl' if model_name != 'neural_network' else f'{model_name}.pth')
        if not os.path.exists(model_path):
            train_model(model_name)
        
        if model_path.endswith('.pkl'):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
        elif model_path.endswith('.pth'):
            model = SimpleNN(X_test.shape[1])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            with torch.no_grad():
                y_pred = (model(torch.tensor(X_test.values, dtype=torch.float32)) >= 0.5).float().numpy()

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=20)
        ax.set_xlabel('Predicted', fontsize=16)
        ax.set_ylabel('Actual', fontsize=16)

    plt.tight_layout(pad=3.0)
    save_path = os.path.join(PLOTS_DIR, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    msg = f"Confusion matrices plot saved to {save_path}"
    print(msg)
    return msg

if __name__ == "__main__":
    prepare_data()
    train_model('logistic_regression')
    generate_feature_importance_subplots()