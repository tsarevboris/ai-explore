import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    classification_report
)

def generate_linear_data(n_samples=100, noise=1.0, seed=42):
    np.random.seed(seed)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + noise * np.random.randn(n_samples, 1)
    return X, y

def prepare_plot_data(X, y, title="Data"):
    plt.scatter(X, y, color='blue', s=10, alpha=0.5, label='Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)

def evaluate_classification_model(name, model, preprocessor, X_train, y_train, X_valid, y_valid):
    full_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # Cross-val on train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring="accuracy").mean()
    cv_f1 = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring="f1").mean()

    # Fit and test
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_valid)

    # Some models support probability; guard for ROC AUC
    if hasattr(full_pipeline.named_steps["model"], "predict_proba"):
        y_proba = full_pipeline.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_proba)
    else:
        # Fallback: decision_function or skip AUC
        if hasattr(full_pipeline.named_steps["model"], "decision_function"):
            y_scores = full_pipeline.decision_function(X_valid)
            y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)
            auc = roc_auc_score(y_valid, y_proba)
        else:
            auc = np.nan

    acc = accuracy_score(y_valid, y_pred)
    prec = precision_score(y_valid, y_pred, zero_division=0)
    rec = recall_score(y_valid, y_pred, zero_division=0)
    f1 = f1_score(y_valid, y_pred, zero_division=0)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print('='*70)
    print(f"Cross-validation:  Acc={cv_acc:.3f} | F1={cv_f1:.3f}")
    print(f"Test metrics:      Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | AUC={auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_valid, y_pred, digits=3))

    # Create figure with confusion matrix and ROC curve side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Confusion matrix on the left
    ConfusionMatrixDisplay.from_predictions(y_valid, y_pred, ax=axes[0], cmap='Blues')
    axes[0].set_title(f'Confusion Matrix')
    axes[0].grid(False)
    
    # ROC curve on the right (if available)
    if not np.isnan(auc):
        if hasattr(full_pipeline.named_steps["model"], "predict_proba"):
            RocCurveDisplay.from_estimator(full_pipeline, X_valid, y_valid, ax=axes[1])
        elif hasattr(full_pipeline.named_steps["model"], "decision_function"):
            RocCurveDisplay.from_estimator(full_pipeline, X_valid, y_valid, ax=axes[1])
        axes[1].set_title(f'ROC Curve (AUC={auc:.3f})')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'ROC curve not available', 
                     ha='center', va='center', fontsize=12)
        axes[1].set_title('ROC Curve')
        axes[1].axis('off')
    
    fig.suptitle(f'{name} - Performance Evaluation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return {
        "name": name,
        "cv_acc": cv_acc,
        "cv_f1": cv_f1,
        "test_acc": acc,
        "test_prec": prec,
        "test_rec": rec,
        "test_f1": f1,
        "test_auc": auc
    }


def compare_classification_models(models, preprocessor, X_train, y_train, X_valid, y_valid):
    """
    Compare multiple classification models and display results in a summary table.
    
    Parameters
    ----------
    models : dict
        Dictionary of model names to model objects, e.g., {"RF": RandomForestClassifier(), ...}
    preprocessor : sklearn transformer
        Preprocessing pipeline to apply before model
    X_train, y_train : array-like
        Training data
    X_valid, y_valid : array-like
        Validation data
        
    Returns
    -------
    results_df : pandas.DataFrame
        Comparison table with metrics for all models
    """
    import pandas as pd
    
    results = []
    
    for name, model in models.items():
        result = evaluate_classification_model(
            name, model, preprocessor, X_train, y_train, X_valid, y_valid
        )
        results.append(result)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("name")
    
    # Sort by test accuracy (descending)
    results_df = results_df.sort_values("test_acc", ascending=False)
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print('='*60)
    print(results_df.to_string())
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # CV Accuracy
    results_df["cv_acc"].plot(kind="barh", ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("Cross-Validation Accuracy")
    axes[0, 0].set_xlabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test Accuracy
    results_df["test_acc"].plot(kind="barh", ax=axes[0, 1], color="lightcoral")
    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].set_xlabel("Accuracy")
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    results_df["test_f1"].plot(kind="barh", ax=axes[1, 0], color="lightgreen")
    axes[1, 0].set_title("Test F1 Score")
    axes[1, 0].set_xlabel("F1 Score")
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    results_df["test_auc"].plot(kind="barh", ax=axes[1, 1], color="gold")
    axes[1, 1].set_title("Test AUC")
    axes[1, 1].set_xlabel("AUC")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Highlight best model
    best_model = results_df["test_acc"].idxmax()
    print(f"\n🏆 Best Model (by Test Accuracy): {best_model}")
    print(f"   Accuracy: {results_df.loc[best_model, 'test_acc']:.3f}")
    print(f"   F1 Score: {results_df.loc[best_model, 'test_f1']:.3f}")
    print(f"   AUC: {results_df.loc[best_model, 'test_auc']:.3f}")
    
    return results_df
