import joblib
import warnings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    roc_curve
)

warnings.filterwarnings('ignore')


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data' / 'processed' / 'final_dataset.csv'
    models_dir = project_root / 'models'

    df_raw = pd.read_csv(data_path)
    target_col = 'survival_status'
    y = df_raw[target_col]
    X_raw = df_raw.drop(columns=[target_col])

    _, X_holdout, _, y_holdout = train_test_split(
        X_raw, y, test_size=0.10, random_state=42, stratify=y
    )

    models_info = {
        'XGBoost': models_dir / 'xgboost_model.pkl',
        'SVM': models_dir / 'modele_svm_bmt.pkl',
        'Random Forest': models_dir / 'rf_model.pkl',
        'LightGBM': models_dir / 'lgbm_model.pkl',
    }

    results = []
    roc_data = []
    report_lines = []

    for name, path in models_info.items():
        if not path.exists():
            continue

        model = joblib.load(path)
        y_pred = model.predict(X_holdout)

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_holdout)[:, 1]
        else:
            y_proba = None

        acc  = accuracy_score(y_holdout, y_pred)
        prec = precision_score(y_holdout, y_pred, zero_division=0)
        rec  = recall_score(y_holdout, y_pred, zero_division=0)
        f1   = f1_score(y_holdout, y_pred, zero_division=0)

        try:
            roc = roc_auc_score(y_holdout, y_proba if y_proba is not None else y_pred)
        except ValueError:
            roc = 0.0

        results.append({
            'Model':     name,
            'Accuracy':  acc,
            'F1-Score':  f1,
            'Precision': prec,
            'Recall':    rec,
            'ROC-AUC':   roc,
        })

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_holdout, y_proba)
            roc_data.append({'name': name, 'fpr': fpr, 'tpr': tpr, 'auc': roc})

        report_lines.append(f"{'=' * 60}")
        report_lines.append(f"Model: {name}")
        report_lines.append(f"{'=' * 60}")
        report_lines.append(f"  Accuracy  : {acc:.4f}")
        report_lines.append(f"  Precision : {prec:.4f}")
        report_lines.append(f"  Recall    : {rec:.4f}")
        report_lines.append(f"  F1-Score  : {f1:.4f}")
        report_lines.append(f"  ROC-AUC   : {roc:.4f}")
        report_lines.append("")
        report_lines.append("Classification Report:")
        report_lines.append(classification_report(y_holdout, y_pred, zero_division=0))

    if not results:
        return

    df_results = pd.DataFrame(results)

    print("| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |")
    print("|---|---|---|---|---|---|")
    for _, row in df_results.iterrows():
        print(
            f"| {row['Model']} | {row['Accuracy']:.4f} | {row['F1-Score']:.4f} | "
            f"{row['Precision']:.4f} | {row['Recall']:.4f} | {row['ROC-AUC']:.4f} |"
        )

    report_path = models_dir / 'final_evaluation_metrics.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    df_melt = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model')
    plt.title('Final Model Comparison')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(models_dir / 'final_model_comparison.png', dpi=300)
    plt.close()

    if roc_data:
        plt.figure(figsize=(9, 7))
        for entry in roc_data:
            plt.plot(entry['fpr'], entry['tpr'], lw=2,
                     label=f"{entry['name']} (AUC = {entry['auc']:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Combined ROC Curve — All Models')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(models_dir / 'combined_roc_curve.png', dpi=300)
        plt.close()


if __name__ == '__main__':
    main()
