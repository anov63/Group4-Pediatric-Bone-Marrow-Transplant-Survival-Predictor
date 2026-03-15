import logging
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

class ModelTrainer:

    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.data_path = self.project_root / 'data' / 'processed' / 'final_dataset.csv'
        self.models_dir = self.project_root / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        self.X_dev = None
        self.X_holdout = None
        self.y_dev = None
        self.y_holdout = None
        self.feature_names = None

    def prepare_data(self):
        self.logger.info("Preparing data with 90/10 holdout split...")

        df = pd.read_csv(self.data_path)
        target_col = 'survival_status'

        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.feature_names = X.columns.tolist()

        self.X_dev, self.X_holdout, self.y_dev, self.y_holdout = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y
        )

        self.logger.info(
            f"Development set: {len(self.X_dev)} samples | "
            f"Holdout set: {len(self.X_holdout)} samples"
        )

    def _evaluate_and_save(self, model_name, pipeline, y_pred, y_pred_proba=None):
        metrics = {
            'Accuracy':  accuracy_score(self.y_holdout, y_pred),
            'Precision': precision_score(self.y_holdout, y_pred, zero_division=0),
            'Recall':    recall_score(self.y_holdout, y_pred, zero_division=0),
            'F1-Score':  f1_score(self.y_holdout, y_pred, zero_division=0),
        }
        if y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(self.y_holdout, y_pred_proba)

        print(f"\n--- {model_name} Holdout Metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {round(v, 4)}")

        cm = confusion_matrix(self.y_holdout, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Prédit Survie', 'Prédit Décès'],
                    yticklabels=['Réel Survie', 'Réel Décès'],
                    annot_kws={'size': 14})
        plt.title(f'Matrice de Confusion - {model_name}')
        safe_name = model_name.lower().replace(' ', '_')
        plt.savefig(self.models_dir / f'{safe_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        return metrics

    def train_xgboost(self):
        self.logger.info("Training XGBoost with 5-Fold CV on development set...")

        neg_count = sum(self.y_dev == 0)
        pos_count = sum(self.y_dev == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(base_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True)
        search.fit(self.X_dev, self.y_dev)

        best_model = search.best_estimator_
        self.logger.info(f"XGBoost best params: {search.best_params_}")

        y_pred = best_model.predict(self.X_holdout)
        y_pred_proba = best_model.predict_proba(self.X_holdout)[:, 1]

        self._evaluate_and_save('XGBoost', best_model, y_pred, y_pred_proba)

        importance = best_model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(15)

        plt.figure(figsize=(12, 8))
        plt.barh(feat_imp['feature'], feat_imp['importance'], color='skyblue')
        plt.title('Top 15 Feature Importances - XGBoost')
        plt.savefig(self.models_dir / 'xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        joblib.dump(best_model, self.models_dir / 'xgboost_model.pkl')

    def train_svm(self):
        self.logger.info("Training SVM with 5-Fold CV on development set...")

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=42))
        ])

        param_grid = {
            'clf__kernel': ['rbf', 'linear'],
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__gamma': ['scale', 'auto', 0.1],
            'clf__class_weight': ['balanced']
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring='balanced_accuracy', 
            n_jobs=-1, 
            refit=True
        )
        search.fit(self.X_dev, self.y_dev)

        best_pipeline = search.best_estimator_
        self.logger.info(f"SVM best params: {search.best_params_}")

        y_pred = best_pipeline.predict(self.X_holdout)
        y_pred_proba = best_pipeline.predict_proba(self.X_holdout)[:, 1]

        self._evaluate_and_save('SVM', best_pipeline, y_pred, y_pred_proba)

        joblib.dump(best_pipeline, self.models_dir / 'modele_svm_bmt.pkl')
    def train_random_forest(self):
        self.logger.info("Training Random Forest with 5-Fold CV on development set...")

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 5],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True)
        search.fit(self.X_dev, self.y_dev)

        best_pipeline = search.best_estimator_
        self.logger.info(f"Random Forest best params: {search.best_params_}")

        y_pred = best_pipeline.predict(self.X_holdout)
        y_pred_proba = best_pipeline.predict_proba(self.X_holdout)[:, 1]

        self._evaluate_and_save('Random Forest', best_pipeline, y_pred, y_pred_proba)

        joblib.dump(best_pipeline, self.models_dir / 'rf_model.pkl')

    def train_lightgbm(self):
        self.logger.info("Training LightGBM Pipeline with 5-Fold CV on development set...")

        continuous_vars = [col for col in self.X_dev.columns if self.X_dev[col].nunique() > 2]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), continuous_vars)
            ],
            remainder='passthrough'
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42, verbose=-1))
        ])

        param_grid = {
            'clf__num_leaves': [20, 31],
            'clf__max_depth': [4, 6],
            'clf__learning_rate': [0.05, 0.1],
            'clf__n_estimators': [100, 200],
            'clf__colsample_bytree': [0.8, 1.0],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True
        )
        search.fit(self.X_dev, self.y_dev)

        best_pipeline = search.best_estimator_
        self.logger.info(f"LightGBM best params: {search.best_params_}")

        y_pred = best_pipeline.predict(self.X_holdout)
        y_pred_proba = best_pipeline.predict_proba(self.X_holdout)[:, 1]

        self._evaluate_and_save('LightGBM', best_pipeline, y_pred, y_pred_proba)

        lgbm_model = best_pipeline.named_steps['clf']
        
        passthrough_vars = [col for col in self.X_dev.columns if col not in continuous_vars]
        feature_names_out = continuous_vars + passthrough_vars

        importance_gain = lgbm_model.booster_.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': feature_names_out,
            'Gain': importance_gain
        }).sort_values(by='Gain', ascending=False).head(10)

        print("\n--- TOP 10 FEATURES (GAIN) ---")
        for _, row in importance_df.iterrows():
            print(f"  {row['Feature']:20s} : {row['Gain']:.2f}")

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Gain', y='Feature', data=importance_df, palette='viridis')
        plt.title('Top 10 Feature Importances (Gain) - LightGBM')
        plt.xlabel('Total Gain')
        plt.tight_layout()
        plt.savefig(self.models_dir / 'lgbm_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

        joblib.dump(best_pipeline, self.models_dir / 'lgbm_model.pkl')

    def run_all(self):
        self.logger.info("Starting all model training processes...")
        self.prepare_data()
        self.train_xgboost()
        self.train_svm()
        self.train_random_forest()
        self.train_lightgbm()
        self.logger.info("All model training processes have finished.")

if __name__ == '__main__':
    ROOT_DIR = Path(__file__).resolve().parent.parent
    trainer = ModelTrainer(project_root=ROOT_DIR)
    trainer.run_all()
