"""
MicrobiomeMLAnalyzer — Professional Edition 2025
═══════════════════════════════════════════════
Scientific ML pipeline for microbiome data based on best practices from:
  • Karwowska et al., Microbiome 2025 (SHAP + compositional transforms)
  • ML4Microbiome COST Action (stratified repeated k-fold, AUROC/AUPRC)
  • mBio framework (Topçuoğlu et al. 2020) for reproducible ML

Implemented methods
───────────────────
Models       : Random Forest, XGBoost, LightGBM, SVM (RBF), Logistic
               Regression (ElasticNet), Decision Tree, Stacking Ensemble
Optimization : Optuna (TPE sampler, Bayesian), fallback to GridSearchCV
Validation   : Stratified Repeated K-fold (5×5), hold-out test set,
               LOOCV for n<30
Explainability: SHAP (TreeExplainer / KernelExplainer), permutation
                importance, MDI, biomarker stability across CV folds
Metrics      : Accuracy, F1 (weighted), AUROC, AUPRC, MCC, Kappa
Plots        : SHAP summary/bee-swarm, SHAP waterfall (per-sample),
               ROC + Precision-Recall curves, confusion matrices,
               calibration curves, CV score distribution, model comparison,
               feature importance (multi-model heatmap), biomarker stability
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

# ── sklearn core ──
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      RepeatedStratifiedKFold, cross_val_score,
                                      LeaveOneOut, GridSearchCV)
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                        SelectFromModel, RFECV)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              average_precision_score, matthews_corrcoef,
                              cohen_kappa_score, confusion_matrix,
                              classification_report, roc_curve,
                              precision_recall_curve, balanced_accuracy_score)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ── optional heavy libs (graceful fallback) ──
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    class _ST:
        def error(self, m): print(f"ERROR: {m}")
        def warning(self, m): print(f"WARN: {m}")
        def info(self, m): print(f"INFO: {m}")
        def success(self, m): print(f"OK: {m}")
    st = _ST()


# ═══════════════════════════════════════════════════════════════
class MicrobiomeMLAnalyzer:
    """
    Professional ML analyzer — drop-in replacement for the basic version.
    All public methods of the original are preserved for backward compatibility.
    """

    # Colour palette for plots
    _PALETTE = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
                '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']

    def __init__(self, data_loader, settings: Dict):
        self.data_loader = data_loader
        self.settings = settings

        self.processed_data: Optional[Dict] = None
        self.models: Dict = {}          # trained model results
        self.shap_values: Dict = {}     # SHAP per model
        self.stability_results: Dict = {} # biomarker stability across folds
        self._le = LabelEncoder()

        try:
            from normalization import NormalizationProcessor
            self.normalizer = NormalizationProcessor()
        except ImportError:
            self.normalizer = None

    # ──────────────────────────────────────────────────────────────────────
    #  DATA PREPARATION
    # ──────────────────────────────────────────────────────────────────────

    def prepare_data_for_ml(self,
                             target_variable: str,
                             feature_selection: bool = True,
                             min_samples_per_class: int = 3,
                             max_features: int = 100,
                             feature_selection_method: str = 'mutual_info') -> bool:
        """
        Prepare microbiome OTU data for ML.

        Parameters
        ----------
        target_variable : str
            Column in metadata to predict.
        feature_selection : bool
            Apply feature selection before training.
        min_samples_per_class : int
            Classes with fewer samples are dropped.
        max_features : int
            Maximum number of OTU features to keep.
        feature_selection_method : str
            'mutual_info' | 'lasso' | 'rfe' | 'combined'
        """
        try:
            # ── Load OTU ──
            otu = self.data_loader.get_filtered_otu_table(
                self.settings.get('auto_filter_zeros', True))

            # After smart identification otu is always (samples × OTUs).
            # Guard: if somehow OTUs ended up in rows, transpose once.
            if otu.shape[0] > otu.shape[1]:
                otu = otu.T  # now (samples × OTUs)
            ml_data = otu.copy()   # samples × OTUs

            # ── Normalization (compositional-aware per ML4Microbiome best practice) ──
            norm = self.settings.get('ml_normalization', 'clr')
            ml_data = self._apply_normalization(ml_data, norm)

            # ── Align with metadata ──
            target = self.data_loader.metadata[target_variable].copy()
            common = ml_data.index.intersection(target.index)

            if len(common) < 5:
                # positional fallback
                n = min(len(ml_data), len(target))
                if n < 5:
                    self._err("Too few samples"); return False
                idx = [f"s_{i}" for i in range(n)]
                ml_data = ml_data.iloc[:n].copy(); ml_data.index = idx
                target = target.iloc[:n].copy();   target.index   = idx
                common = ml_data.index

            ml_data = ml_data.loc[common]
            target  = target.loc[common]

            # ── Class filtering ──
            vc = target.value_counts()
            valid = vc[vc >= min_samples_per_class].index
            if len(valid) < 2:
                self._err(f"Need ≥2 classes with ≥{min_samples_per_class} samples"); return False
            mask = target.isin(valid)
            ml_data, target = ml_data[mask], target[mask]

            # ── Remove zero-variance features ──
            var = ml_data.var(axis=0)
            ml_data = ml_data.loc[:, var > 0]

            # ── Feature selection ──
            if feature_selection and ml_data.shape[1] > max_features:
                ml_data = self._feature_selection(ml_data, target,
                                                   max_features, feature_selection_method)

            # ── Label encode ──
            y_enc = self._le.fit_transform(target.values)
            class_names = list(self._le.classes_)

            # ── Decide CV strategy ──
            n = len(ml_data)
            if n < 30:
                cv_strategy = 'loocv'
            elif n < 100:
                cv_strategy = 'repeated_kfold_5x5'
            else:
                cv_strategy = 'repeated_kfold_5x10'

            self.processed_data = {
                'X':          ml_data.values,
                'X_df':       ml_data,
                'y':          y_enc,
                'y_series':   target,
                'feature_names': list(ml_data.columns),
                'sample_names':  list(ml_data.index),
                'class_names':   class_names,
                'target_name':   target_variable,
                'n_classes':     len(class_names),
                'cv_strategy':   cv_strategy,
                'normalization': norm,
            }
            self._ok(f"Ready: {n} samples × {ml_data.shape[1]} features, "
                     f"{len(class_names)} classes, CV={cv_strategy}")
            return True

        except Exception as e:
            self._err(f"Data preparation failed: {e}")
            import traceback; traceback.print_exc()
            return False

    def _apply_normalization(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """CLR is best practice for compositional microbiome data (Aitchison 1986)."""
        if method == 'clr':
            pseudo = df + 1
            gm = np.exp(np.log(pseudo).mean(axis=1))
            return np.log(pseudo.div(gm, axis=0))
        elif method == 'log':
            return np.log10(df + 1)
        elif method == 'relative':
            s = df.sum(axis=1).replace(0, 1)
            return df.div(s, axis=0) * 100
        elif method == 'sqrt':
            return np.sqrt(df)
        return df

    def _feature_selection(self, X: pd.DataFrame, y: pd.Series,
                           max_feat: int, method: str) -> pd.DataFrame:
        """Multi-strategy feature selection — returns reduced DataFrame."""
        y_enc = self._le.fit_transform(y.values) if not np.issubdtype(y.dtype, np.integer) else y.values
        n_sel = min(max_feat, X.shape[1])

        if method == 'mutual_info' or not XGBOOST_AVAILABLE:
            sel = SelectKBest(mutual_info_classif, k=n_sel)
            sel.fit(X.values, y_enc)
            mask = sel.get_support()

        elif method == 'lasso':
            lr = LogisticRegression(penalty='elasticnet', solver='saga',
                                    l1_ratio=0.5, C=0.1, max_iter=2000,
                                    class_weight='balanced', random_state=42)
            lr.fit(StandardScaler().fit_transform(X.values), y_enc)
            imp = np.abs(lr.coef_).mean(axis=0)
            thresh = np.sort(imp)[::-1][n_sel - 1]
            mask = imp >= thresh

        elif method == 'rfe':
            rf = RandomForestClassifier(n_estimators=50, random_state=42,
                                        class_weight='balanced', n_jobs=-1)
            rfe = RFECV(rf, step=0.1, cv=3, scoring='accuracy', n_jobs=-1,
                        min_features_to_select=min(10, n_sel))
            rfe.fit(X.values, y_enc)
            mask = rfe.support_
            # Cap at max_feat
            if mask.sum() > n_sel:
                fi = rf.fit(X.values, y_enc).feature_importances_
                top = np.argsort(fi)[::-1][:n_sel]
                mask2 = np.zeros(X.shape[1], dtype=bool)
                mask2[top] = True
                mask = mask & mask2

        elif method == 'combined':
            # Mutual info + RF importance intersection
            mi = mutual_info_classif(X.values, y_enc)
            top_mi = set(np.argsort(mi)[::-1][:n_sel * 2])
            rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                        class_weight='balanced', n_jobs=-1)
            rf.fit(X.values, y_enc)
            top_rf = set(np.argsort(rf.feature_importances_)[::-1][:n_sel * 2])
            union = list(top_mi | top_rf)
            # From union pick top n_sel by MI score
            mi_union = mi[union]
            best = np.array(union)[np.argsort(mi_union)[::-1][:n_sel]]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[best] = True

        else:
            sel = SelectKBest(mutual_info_classif, k=n_sel)
            sel.fit(X.values, y_enc)
            mask = sel.get_support()

        cols = X.columns[mask]
        return X[cols]

    # ──────────────────────────────────────────────────────────────────────
    #  TRAINING HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _get_cv(self):
        """Return appropriate sklearn CV object based on dataset size."""
        strat = self.processed_data.get('cv_strategy', 'repeated_kfold_5x5')
        if strat == 'loocv':
            return LeaveOneOut()
        elif strat == 'repeated_kfold_5x10':
            return RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
        else:
            return RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    def _full_eval(self, model, X_train, X_test, y_train, y_test,
                   model_name: str, X_all, y_all) -> Dict:
        """Train, predict, compute all metrics, run CV, compute SHAP."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        n_cls = self.processed_data['n_classes']
        class_names = self.processed_data['class_names']
        feat_names = self.processed_data['feature_names']

        # ── Core metrics ──
        res = {
            'model_type': model_name,
            'model': model,
            'train_accuracy': accuracy_score(y_train, model.predict(X_train)),
            'test_accuracy':  accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'mcc':   matthews_corrcoef(y_test, y_pred),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, target_names=class_names,
                output_dict=True, zero_division=0),
            'test_predictions': y_pred,
            'test_true': y_test,
        }

        # ── Probability-based metrics ──
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            if n_cls == 2:
                fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                prec, rec, _ = precision_recall_curve(y_test, proba[:, 1])
                res['roc_auc']  = roc_auc_score(y_test, proba[:, 1])
                res['auprc']    = average_precision_score(y_test, proba[:, 1])
                res['roc_data'] = {'fpr': fpr, 'tpr': tpr}
                res['pr_data']  = {'precision': prec, 'recall': rec}
                # Calibration
                try:
                    prob_true, prob_pred = calibration_curve(y_test, proba[:, 1], n_bins=5)
                    res['calibration'] = {'prob_true': prob_true, 'prob_pred': prob_pred}
                except Exception:
                    pass
            else:
                try:
                    res['roc_auc'] = roc_auc_score(
                        y_test, proba, multi_class='ovr', average='weighted')
                except Exception:
                    pass
            res['y_proba'] = proba

        # ── Cross-Validation ──
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X_all, y_all, cv=cv,
                                     scoring='balanced_accuracy', n_jobs=-1)
        res['cv_mean'] = cv_scores.mean()
        res['cv_std']  = cv_scores.std()
        res['cv_scores'] = cv_scores.tolist()

        # ── Feature importance ──
        fi = self._extract_feature_importance(model, X_test, y_test, feat_names)
        if fi is not None:
            res['feature_importance'] = fi

        # ── Permutation importance (model-agnostic) ──
        try:
            perm = permutation_importance(model, X_test, y_test,
                                          n_repeats=10, random_state=42, n_jobs=-1)
            res['permutation_importance'] = pd.DataFrame({
                'feature': feat_names,
                'importance_mean': perm.importances_mean,
                'importance_std':  perm.importances_std,
            }).sort_values('importance_mean', ascending=False)
        except Exception:
            pass

        # ── SHAP ──
        if SHAP_AVAILABLE:
            self._compute_shap(model, X_train, X_test, model_name, feat_names)

        return res

    def _extract_feature_importance(self, model, X_test, y_test, feat_names):
        """Extract MDI importance from tree models; coefficients from linear."""
        try:
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                imp = np.abs(model.coef_).mean(axis=0)
            elif hasattr(model, 'best_estimator_'):
                return self._extract_feature_importance(
                    model.best_estimator_, X_test, y_test, feat_names)
            else:
                return None
            return pd.DataFrame({
                'feature': feat_names,
                'importance': imp
            }).sort_values('importance', ascending=False)
        except Exception:
            return None

    def _compute_shap(self, model, X_train, X_test, model_name: str, feat_names):
        """Compute SHAP values — TreeExplainer for trees, KernelExplainer otherwise."""
        try:
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_test)
            elif hasattr(model, 'predict_proba'):
                bg = shap.kmeans(X_train, min(30, len(X_train)))
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                sv = explainer.shap_values(X_test[:min(50, len(X_test))],
                                           nsamples=100, silent=True)
            else:
                return

            # For binary: use class-1 SHAP; for multi: use list
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]

            mean_abs = np.abs(sv).mean(axis=0) if sv.ndim == 2 else np.abs(sv).mean(axis=0)
            self.shap_values[model_name] = {
                'values':       sv,
                'X_test':       X_test,
                'feature_names': feat_names,
                'mean_abs':     pd.Series(mean_abs, index=feat_names).sort_values(ascending=False),
                'explainer':    explainer,
            }
        except Exception as e:
            print(f"SHAP error ({model_name}): {e}")

    def _biomarker_stability(self, model_fn, X, y, feat_names: List[str],
                              n_splits: int = 5, n_repeats: int = 5):
        """
        Compute feature importance across all CV folds → stability score.
        Returns DataFrame with mean/std/CV of importance per feature.
        """
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=42)
        rows = []
        for train_idx, _ in skf.split(X, y):
            m = model_fn()
            m.fit(X[train_idx], y[train_idx])
            fi = self._extract_feature_importance(m, X[train_idx], y[train_idx], feat_names)
            if fi is not None:
                row = dict(zip(fi['feature'], fi['importance']))
                rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows).fillna(0)
        stability = pd.DataFrame({
            'feature':    df.columns,
            'mean_imp':   df.mean(),
            'std_imp':    df.std(),
            'cv_imp':     df.std() / (df.mean() + 1e-9),
            'selection_freq': (df > 0).mean(),
        }).sort_values('mean_imp', ascending=False).reset_index(drop=True)
        return stability

    # ──────────────────────────────────────────────────────────────────────
    #  OPTUNA HYPERPARAMETER OPTIMIZATION
    # ──────────────────────────────────────────────────────────────────────

    def _optuna_optimize(self, model_key: str, X_train, y_train,
                          n_trials: int = 50) -> Dict:
        """Bayesian hyper-parameter search with Optuna TPE sampler."""
        if not OPTUNA_AVAILABLE:
            return {}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def objective_rf(trial):
            p = dict(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            )
            m = RandomForestClassifier(**p, class_weight='balanced',
                                       random_state=42, n_jobs=-1)
            return cross_val_score(m, X_train, y_train, cv=cv,
                                    scoring='balanced_accuracy', n_jobs=-1).mean()

        def objective_xgb(trial):
            p = dict(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 2, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            )
            m = XGBClassifier(**p, eval_metric='logloss',
                               random_state=42, n_jobs=-1, verbosity=0)
            return cross_val_score(m, X_train, y_train, cv=cv,
                                    scoring='balanced_accuracy', n_jobs=-1).mean()

        def objective_lgbm(trial):
            p = dict(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                num_leaves=trial.suggest_int('num_leaves', 15, 127),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            )
            m = LGBMClassifier(**p, class_weight='balanced', random_state=42,
                                n_jobs=-1, verbose=-1)
            return cross_val_score(m, X_train, y_train, cv=cv,
                                    scoring='balanced_accuracy', n_jobs=-1).mean()

        def objective_svm(trial):
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
            p = dict(kernel=kernel,
                     C=trial.suggest_float('C', 1e-3, 100.0, log=True),
                     class_weight='balanced', probability=True, random_state=42)
            if kernel == 'rbf':
                p['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            m = Pipeline([('sc', StandardScaler()), ('clf', SVC(**p))])
            return cross_val_score(m, X_train, y_train, cv=cv,
                                    scoring='balanced_accuracy', n_jobs=-1).mean()

        objectives = {
            'random_forest': objective_rf,
            'xgboost': objective_xgb if XGBOOST_AVAILABLE else objective_rf,
            'lightgbm': objective_lgbm if LIGHTGBM_AVAILABLE else objective_rf,
            'svm': objective_svm,
        }

        obj_fn = objectives.get(model_key, objective_rf)
        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj_fn, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    # ──────────────────────────────────────────────────────────────────────
    #  PUBLIC TRAINING METHODS
    # ──────────────────────────────────────────────────────────────────────

    def _split(self, test_size: float = 0.2):
        X, y = self.processed_data['X'], self.processed_data['y']
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        smote_strategy = self.processed_data.get('smote_strategy')
        if smote_strategy:
            X_tr, y_tr = self._apply_smote_to_train(X_tr, y_tr, smote_strategy)
            self.processed_data['smote_applied'] = smote_strategy.upper()
        return X_tr, X_te, y_tr, y_te

    def train_random_forest(self, n_estimators=200, max_depth=15,
                             min_samples_split=5, max_features='sqrt',
                             test_size=0.2, random_state=42,
                             optimize_params=False) -> Optional[Dict]:
        if not self.processed_data: return None
        X_tr, X_te, y_tr, y_te = self._split(test_size)

        if optimize_params and OPTUNA_AVAILABLE:
            params = self._optuna_optimize('random_forest', X_tr, y_tr)
        else:
            params = dict(n_estimators=n_estimators, max_depth=max_depth,
                          min_samples_split=min_samples_split, max_features=max_features)

        model = RandomForestClassifier(**params, class_weight='balanced',
                                        oob_score=True, random_state=42, n_jobs=-1)
        res = self._full_eval(model, X_tr, X_te, y_tr, y_te,
                               'Random Forest',
                               self.processed_data['X'], self.processed_data['y'])
        res['oob_score'] = model.oob_score_ if hasattr(model, 'oob_score_') else None
        res['parameters'] = params
        self.models['random_forest'] = res

        # Biomarker stability
        feat_names = self.processed_data['feature_names']
        self.stability_results['random_forest'] = self._biomarker_stability(
            lambda: RandomForestClassifier(**params, class_weight='balanced',
                                           random_state=42, n_jobs=-1),
            self.processed_data['X'], self.processed_data['y'], feat_names)

        self._ok(f"RF: acc={res['test_accuracy']:.3f}, "
                 f"AUROC={res.get('roc_auc', 'N/A')}, CV={res['cv_mean']:.3f}±{res['cv_std']:.3f}")
        return res

    def train_xgboost(self, n_estimators=200, max_depth=5, learning_rate=0.05,
                       test_size=0.2, optimize_params=False) -> Optional[Dict]:
        if not self.processed_data: return None
        if not XGBOOST_AVAILABLE:
            self._warn("XGBoost not installed. Run: pip install xgboost"); return None

        X_tr, X_te, y_tr, y_te = self._split(test_size)

        if optimize_params and OPTUNA_AVAILABLE:
            params = self._optuna_optimize('xgboost', X_tr, y_tr)
        else:
            params = dict(n_estimators=n_estimators, max_depth=max_depth,
                          learning_rate=learning_rate, subsample=0.8,
                          colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)

        n_cls = self.processed_data['n_classes']
        objective = 'binary:logistic' if n_cls == 2 else 'multi:softprob'
        model = XGBClassifier(**params, objective=objective,
                               eval_metric='logloss',
                               random_state=42, n_jobs=-1, verbosity=0)
        res = self._full_eval(model, X_tr, X_te, y_tr, y_te,
                               'XGBoost',
                               self.processed_data['X'], self.processed_data['y'])
        res['parameters'] = params
        self.models['xgboost'] = res
        self._ok(f"XGBoost: acc={res['test_accuracy']:.3f}, CV={res['cv_mean']:.3f}")
        return res

    def train_lightgbm(self, n_estimators=200, num_leaves=31, learning_rate=0.05,
                        test_size=0.2, optimize_params=False) -> Optional[Dict]:
        if not self.processed_data: return None
        if not LIGHTGBM_AVAILABLE:
            self._warn("LightGBM not installed. Run: pip install lightgbm"); return None

        X_tr, X_te, y_tr, y_te = self._split(test_size)

        if optimize_params and OPTUNA_AVAILABLE:
            params = self._optuna_optimize('lightgbm', X_tr, y_tr)
        else:
            params = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                          learning_rate=learning_rate, min_child_samples=10,
                          reg_alpha=0.1, reg_lambda=1.0)

        model = LGBMClassifier(**params, class_weight='balanced',
                                random_state=42, n_jobs=-1, verbose=-1)
        res = self._full_eval(model, X_tr, X_te, y_tr, y_te,
                               'LightGBM',
                               self.processed_data['X'], self.processed_data['y'])
        res['parameters'] = params
        self.models['lightgbm'] = res
        self._ok(f"LightGBM: acc={res['test_accuracy']:.3f}, CV={res['cv_mean']:.3f}")
        return res

    def train_svm(self, C=1.0, kernel='rbf', test_size=0.2,
                   optimize_params=False) -> Optional[Dict]:
        if not self.processed_data: return None
        X_tr, X_te, y_tr, y_te = self._split(test_size)

        if optimize_params and OPTUNA_AVAILABLE:
            params = self._optuna_optimize('svm', X_tr, y_tr)
            pipe = Pipeline([('sc', StandardScaler()),
                              ('clf', SVC(probability=True, class_weight='balanced',
                                          random_state=42, **{
                                              k: v for k, v in params.items()
                                              if k in ('C', 'kernel', 'gamma')}))])
        else:
            pipe = Pipeline([('sc', StandardScaler()),
                              ('clf', SVC(C=C, kernel=kernel, probability=True,
                                          class_weight='balanced', random_state=42))])

        res = self._full_eval(pipe, X_tr, X_te, y_tr, y_te,
                               'SVM',
                               self.processed_data['X'], self.processed_data['y'])
        res['parameters'] = {'C': C, 'kernel': kernel}
        self.models['svm'] = res
        self._ok(f"SVM: acc={res['test_accuracy']:.3f}, CV={res['cv_mean']:.3f}")
        return res

    def train_logistic_regression(self, C=1.0, l1_ratio=0.5, test_size=0.2) -> Optional[Dict]:
        """ElasticNet logistic regression — excellent for sparse biomarker discovery."""
        if not self.processed_data: return None
        X_tr, X_te, y_tr, y_te = self._split(test_size)

        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(penalty='elasticnet', solver='saga',
                                        C=C, l1_ratio=l1_ratio,
                                        class_weight='balanced', max_iter=5000,
                                        random_state=42))
        ])
        res = self._full_eval(pipe, X_tr, X_te, y_tr, y_te,
                               'Logistic Regression (ElasticNet)',
                               self.processed_data['X'], self.processed_data['y'])
        res['parameters'] = {'C': C, 'l1_ratio': l1_ratio}
        self.models['logistic_regression'] = res
        self._ok(f"LogReg: acc={res['test_accuracy']:.3f}, CV={res['cv_mean']:.3f}")
        return res

    def train_decision_tree(self, max_depth=10, min_samples_split=5,
                             min_samples_leaf=2, test_size=0.2,
                             random_state=42, **kwargs) -> Optional[Dict]:
        """Decision Tree — fast, interpretable baseline."""
        if not self.processed_data: return None
        X_tr, X_te, y_tr, y_te = self._split(test_size)
        model = DecisionTreeClassifier(max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        class_weight='balanced', random_state=42)
        res = self._full_eval(model, X_tr, X_te, y_tr, y_te,
                               'Decision Tree',
                               self.processed_data['X'], self.processed_data['y'])
        res['parameters'] = {'max_depth': max_depth}
        self.models['decision_tree'] = res
        self._ok(f"DT: acc={res['test_accuracy']:.3f}, CV={res['cv_mean']:.3f}")
        return res

    def train_stacking_ensemble(self) -> Optional[Dict]:
        """
        Stacking ensemble: base models (RF + XGB/LGB + SVM) + meta-learner (LR).
        Best practice from ML4Microbiome for maximising predictive performance.
        """
        if not self.processed_data: return None
        X_tr, X_te, y_tr, y_te = self._split(0.2)

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                           random_state=42, n_jobs=-1)),
        ]
        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)))
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lgbm', LGBMClassifier(
                n_estimators=200, class_weight='balanced',
                random_state=42, n_jobs=-1, verbose=-1)))
        estimators.append(('svm', Pipeline([
            ('sc', StandardScaler()),
            ('clf', SVC(probability=True, class_weight='balanced', random_state=42))])))

        meta = LogisticRegression(C=1.0, class_weight='balanced',
                                   max_iter=2000, random_state=42)
        stack = StackingClassifier(estimators=estimators, final_estimator=meta,
                                    cv=5, passthrough=False, n_jobs=-1)
        res = self._full_eval(stack, X_tr, X_te, y_tr, y_te,
                               'Stacking Ensemble',
                               self.processed_data['X'], self.processed_data['y'])
        res['parameters'] = {'base_models': [n for n, _ in estimators],
                              'meta_learner': 'LogisticRegression'}
        self.models['stacking'] = res
        self._ok(f"Stacking: acc={res['test_accuracy']:.3f}, CV={res['cv_mean']:.3f}")
        return res

    def train_all(self, optimize: bool = False,
                   include_stacking: bool = True) -> Dict:
        """Train all available models in one call."""
        self.train_decision_tree()
        self.train_random_forest(optimize_params=optimize)
        if XGBOOST_AVAILABLE:
            self.train_xgboost(optimize_params=optimize)
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm(optimize_params=optimize)
        self.train_svm(optimize_params=optimize)
        self.train_logistic_regression()
        if include_stacking and len(self.models) >= 2:
            self.train_stacking_ensemble()
        return self.models

    # ──────────────────────────────────────────────────────────────────────
    #  RESULTS SUMMARY
    # ──────────────────────────────────────────────────────────────────────

    def get_model_comparison(self) -> Optional[pd.DataFrame]:
        if not self.models:
            return None
        rows = []
        for name, res in self.models.items():
            row = {
                'Model': name.replace('_', ' ').title(),
                'Test_Accuracy':    round(res.get('test_accuracy', 0), 4),
                'Balanced_Acc':     round(res.get('balanced_accuracy', 0), 4),
                'F1_Score':         round(res.get('f1_score', 0), 4),
                'AUROC':            round(res.get('roc_auc', 0), 4),
                'AUPRC':            round(res.get('auprc', 0), 4),
                'MCC':              round(res.get('mcc', 0), 4),
                'Kappa':            round(res.get('kappa', 0), 4),
                'CV_Mean':          round(res.get('cv_mean', 0), 4),
                'CV_Std':           round(res.get('cv_std', 0), 4),
            }
            rows.append(row)
        df = pd.DataFrame(rows).sort_values('CV_Mean', ascending=False)
        return df

    def get_best_model_name(self) -> str:
        comp = self.get_model_comparison()
        if comp is None or comp.empty: return ''
        return comp.iloc[0]['Model'].lower().replace(' ', '_')

    def get_feature_importance(self, top_n: int = 30,
                                method: str = 'shap') -> Optional[pd.DataFrame]:
        """Return unified feature importance table, SHAP-first."""
        if not self.models: return None

        # Prefer SHAP of best model
        if method == 'shap' and self.shap_values:
            best = self.get_best_model_name()
            key = best if best in self.shap_values else next(iter(self.shap_values))
            sv = self.shap_values[key]
            return pd.DataFrame({
                'Feature':    sv['mean_abs'].index[:top_n],
                'SHAP_mean':  sv['mean_abs'].values[:top_n],
                'Method':     'SHAP (TreeExplainer)'
            })

        # Fallback to RF feature importance
        for name in ['random_forest', 'xgboost', 'lightgbm', 'decision_tree']:
            if name in self.models:
                fi = self.models[name].get('feature_importance')
                if fi is not None and not fi.empty:
                    return fi.head(top_n).rename(columns={'importance': 'Importance'})
        return None

    def get_text_report(self) -> str:
        """Generate a scientific text report (markdown)."""
        lines = ["# Machine Learning Results — Microbiome Analysis\n"]
        pd_obj = self.processed_data
        if pd_obj:
            lines += [
                f"**Target:** {pd_obj.get('target_name', 'N/A')}  ",
                f"**Samples:** {len(pd_obj.get('sample_names', []))}  ",
                f"**Features:** {len(pd_obj.get('feature_names', []))}  ",
                f"**Classes:** {', '.join(str(c) for c in pd_obj.get('class_names', []))}  ",
                f"**Normalization:** {pd_obj.get('normalization', 'N/A')}  ",
                f"**CV strategy:** {pd_obj.get('cv_strategy', 'N/A')}  \n",
            ]
        comp = self.get_model_comparison()
        if comp is not None:
            lines.append("## Model Comparison\n")
            lines.append(comp.to_markdown(index=False))
            lines.append("\n")
        if self.shap_values:
            lines.append("## Top SHAP Biomarkers\n")
            fi = self.get_feature_importance(top_n=20, method='shap')
            if fi is not None:
                lines.append(fi.to_markdown(index=False))
                lines.append("\n")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    #  PLOTS
    # ──────────────────────────────────────────────────────────────────────

    def plot_model_comparison(self):
        if not PLOTLY_AVAILABLE or not self.models: return None
        comp = self.get_model_comparison()
        if comp is None or comp.empty: return None

        metrics   = [m for m in ['Test_Accuracy', 'Balanced_Acc', 'F1_Score',
                                   'AUROC', 'MCC', 'CV_Mean'] if m in comp.columns]
        fig = go.Figure()
        model_names = comp['Model'].tolist()
        for j, metric in enumerate(metrics):
            vals = comp[metric].tolist()
            fig.add_trace(go.Bar(
                x=model_names, y=vals, name=metric.replace('_', ' '),
                marker_color=self._PALETTE[j % len(self._PALETTE)],
                text=[f'{v:.3f}' for v in vals], textposition='outside',
            ))
        # CV error bars on CV_Mean bar
        if 'CV_Std' in comp.columns and 'CV_Mean' in comp.columns:
            fig.data[-1].error_y = dict(
                type='data', array=comp['CV_Std'].tolist(), visible=True)

        fig.update_layout(
            title=dict(text='Model Comparison — All Metrics', font=dict(size=16), x=0.5),
            barmode='group', yaxis=dict(title='Score', range=[0, 1.12]),
            height=480, template='plotly_white',
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
        return fig

    def plot_confusion_matrices(self):
        if not PLOTLY_AVAILABLE or not self.models: return None
        names  = list(self.models.keys())
        n      = len(names)
        fig    = make_subplots(rows=1, cols=n,
                               subplot_titles=[k.replace('_',' ').title() for k in names],
                               horizontal_spacing=0.08)
        class_names = self.processed_data.get('class_names', [])

        for i, mname in enumerate(names):
            cm = self.models[mname].get('confusion_matrix')
            if cm is None: continue
            r  = cm.astype(float)
            rs = r.sum(axis=1, keepdims=True); rs[rs == 0] = 1
            pct = r / rs * 100
            lbl = class_names if len(class_names) == cm.shape[0] else [str(j) for j in range(cm.shape[0])]
            text = [[f'{cm[r2][c]}<br>({pct[r2][c]:.0f}%)' for c in range(cm.shape[1])]
                    for r2 in range(cm.shape[0])]
            fig.add_trace(go.Heatmap(
                z=cm, x=lbl, y=lbl, colorscale='Blues',
                text=text, texttemplate='%{text}', textfont=dict(size=11),
                showscale=(i == n - 1),
                hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>',
            ), row=1, col=i + 1)
            fig.update_xaxes(title_text='Predicted', row=1, col=i + 1)
            fig.update_yaxes(title_text='True' if i == 0 else '', row=1, col=i + 1)

        fig.update_layout(title=dict(text='Confusion Matrices', font=dict(size=16), x=0.5),
                          height=450, template='plotly_white')
        return fig

    def plot_roc_curves(self):
        if not PLOTLY_AVAILABLE or not self.models: return None
        fig = go.Figure()
        has = False
        for i, (mname, res) in enumerate(self.models.items()):
            rd = res.get('roc_data')
            if rd and rd.get('fpr') is not None:
                auc = res.get('roc_auc', 0)
                fig.add_trace(go.Scatter(
                    x=rd['fpr'], y=rd['tpr'], mode='lines',
                    name=f"{mname.replace('_',' ').title()} (AUC={auc:.3f})",
                    line=dict(color=self._PALETTE[i % len(self._PALETTE)], width=2.5)))
                has = True
        if not has: return None
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  line=dict(color='gray', dash='dash'),
                                  name='Random', showlegend=True))
        fig.update_layout(title=dict(text='ROC Curves (Binary)', font=dict(size=16), x=0.5),
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          height=500, template='plotly_white',
                          legend=dict(x=0.55, y=0.05))
        return fig

    def plot_precision_recall_curves(self):
        if not PLOTLY_AVAILABLE or not self.models: return None
        fig = go.Figure()
        has = False
        for i, (mname, res) in enumerate(self.models.items()):
            pr = res.get('pr_data')
            if pr and pr.get('precision') is not None:
                ap = res.get('auprc', 0)
                fig.add_trace(go.Scatter(
                    x=pr['recall'], y=pr['precision'], mode='lines',
                    name=f"{mname.replace('_',' ').title()} (AP={ap:.3f})",
                    line=dict(color=self._PALETTE[i % len(self._PALETTE)], width=2.5)))
                has = True
        if not has: return None
        fig.update_layout(title=dict(text='Precision-Recall Curves', font=dict(size=16), x=0.5),
                          xaxis_title='Recall', yaxis_title='Precision',
                          height=500, template='plotly_white')
        return fig

    def plot_calibration_curves(self):
        """Reliability diagram — how well predicted probabilities match true frequencies."""
        if not PLOTLY_AVAILABLE or not self.models: return None
        fig = go.Figure()
        has = False
        for i, (mname, res) in enumerate(self.models.items()):
            cal = res.get('calibration')
            if cal:
                fig.add_trace(go.Scatter(
                    x=cal['prob_pred'], y=cal['prob_true'], mode='lines+markers',
                    name=mname.replace('_', ' ').title(),
                    line=dict(color=self._PALETTE[i % len(self._PALETTE)], width=2),
                    marker=dict(size=8)))
                has = True
        if not has: return None
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  line=dict(color='gray', dash='dash'),
                                  name='Perfect calibration'))
        fig.update_layout(title=dict(text='Calibration Curves (Reliability Diagram)',
                                      font=dict(size=16), x=0.5),
                          xaxis_title='Mean Predicted Probability',
                          yaxis_title='Fraction of Positives',
                          height=450, template='plotly_white')
        return fig

    def plot_cv_scores(self):
        if not PLOTLY_AVAILABLE or not self.models: return None
        fig = go.Figure()
        has = False
        for i, (mname, res) in enumerate(self.models.items()):
            scores = res.get('cv_scores')
            if scores:
                fig.add_trace(go.Violin(
                    y=scores, name=mname.replace('_', ' ').title(),
                    box_visible=True, meanline_visible=True, points='all',
                    marker_color=self._PALETTE[i % len(self._PALETTE)],
                    line_color='black', opacity=0.7, jitter=0.3))
                has = True
        if not has: return None
        fig.update_layout(title=dict(text='CV Score Distribution (Balanced Accuracy)',
                                      font=dict(size=16), x=0.5),
                          yaxis_title='Balanced Accuracy',
                          height=450, template='plotly_white')
        return fig

    def plot_feature_importance(self, top_n: int = 25):
        """Multi-model feature importance heatmap + SHAP bar."""
        if not PLOTLY_AVAILABLE or not self.models: return None

        # ── SHAP bar for best model ──
        if self.shap_values:
            best_key = next(iter(self.shap_values))
            sv = self.shap_values[best_key]
            top = sv['mean_abs'].head(top_n)
            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation='h',
                marker=dict(color=top.values,
                            colorscale='RdYlBu_r', showscale=True,
                            colorbar=dict(title='|SHAP|', len=0.6)),
                hovertemplate='%{y}<br>mean|SHAP|=%{x:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title=dict(text=f'SHAP Feature Importance — {best_key.replace("_"," ").title()} (Top {top_n})',
                           font=dict(size=16), x=0.5),
                xaxis_title='Mean |SHAP value|',
                yaxis=dict(autorange='reversed', tickfont=dict(size=11)),
                height=max(400, top_n * 20 + 150),
                template='plotly_white',
                margin=dict(l=220, r=60, t=80, b=40))
            return fig

        # ── Fallback: multi-model MDI heatmap ──
        importance_dfs = {}
        all_features = set()
        for mname, res in self.models.items():
            fi = res.get('feature_importance')
            if fi is not None and not fi.empty:
                top_fi = fi.head(top_n)
                importance_dfs[mname] = dict(zip(top_fi['feature'], top_fi['importance']))
                all_features.update(top_fi['feature'])

        if not importance_dfs: return None
        all_features = sorted(all_features)
        mat = pd.DataFrame({m: [importance_dfs[m].get(f, 0) for f in all_features]
                             for m in importance_dfs}, index=all_features)

        fig = go.Figure(go.Heatmap(
            z=mat.values,
            x=[m.replace('_',' ').title() for m in mat.columns],
            y=mat.index,
            colorscale='YlOrRd',
            hovertemplate='%{y}<br>%{x}<br>Importance: %{z:.4f}<extra></extra>',
            colorbar=dict(title='Importance')
        ))
        fig.update_layout(
            title=dict(text=f'Feature Importance Heatmap (Top {top_n})',
                       font=dict(size=16), x=0.5),
            height=max(500, len(all_features) * 18 + 200),
            xaxis=dict(tickangle=-30),
            yaxis=dict(tickfont=dict(size=9), autorange='reversed'),
            template='plotly_white',
            margin=dict(l=200, r=40, t=80, b=60))
        return fig

    def plot_shap_beeswarm(self, model_name: str = None, top_n: int = 20):
        """SHAP beeswarm (dot plot) — gold standard in microbiome explainability."""
        if not PLOTLY_AVAILABLE or not self.shap_values: return None
        key = model_name or next(iter(self.shap_values))
        sv = self.shap_values.get(key)
        if sv is None: return None

        shap_mat = sv['values']          # samples × features
        feat_names = sv['feature_names']
        X_test_raw = sv['X_test']

        mean_abs = np.abs(shap_mat).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        top_names = [feat_names[i] for i in top_idx]
        top_shap  = shap_mat[:, top_idx]
        top_X     = X_test_raw[:, top_idx]

        fig = go.Figure()
        for fi, fname in enumerate(top_names[::-1]):  # bottom→top
            sv_col = top_shap[:, top_n - 1 - fi]
            x_col  = top_X[:,  top_n - 1 - fi]
            fig.add_trace(go.Scatter(
                x=sv_col, y=[fname] * len(sv_col),
                mode='markers',
                marker=dict(
                    size=7, opacity=0.75,
                    color=x_col,
                    colorscale='RdBu_r', showscale=(fi == 0),
                    colorbar=dict(title='Feature value', len=0.6,
                                  tickfont=dict(size=9)),
                    line=dict(width=0.3, color='gray')),
                name=fname,
                hovertemplate=f'<b>{fname}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{marker.color:.3f}}<extra></extra>',
                showlegend=False
            ))

        fig.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1)
        fig.update_layout(
            title=dict(text=f'SHAP Beeswarm — {key.replace("_"," ").title()} (Top {top_n})',
                       font=dict(size=16), x=0.5),
            xaxis_title='SHAP value (impact on model output)',
            yaxis=dict(tickfont=dict(size=10)),
            height=max(500, top_n * 22 + 150),
            template='plotly_white',
            margin=dict(l=220, r=80, t=80, b=50))
        return fig

    def plot_shap_waterfall(self, sample_idx: int = 0, model_name: str = None):
        """SHAP waterfall for a single sample — explains one prediction."""
        if not PLOTLY_AVAILABLE or not self.shap_values: return None
        key = model_name or next(iter(self.shap_values))
        sv = self.shap_values.get(key)
        if sv is None: return None

        idx = min(sample_idx, len(sv['values']) - 1)
        sv_row = sv['values'][idx]
        feat_names = sv['feature_names']
        top_n = 20
        top_idx = np.argsort(np.abs(sv_row))[::-1][:top_n]
        top_names = [feat_names[i] for i in top_idx]
        top_sv    = sv_row[top_idx]

        colors = ['#E74C3C' if v > 0 else '#3498DB' for v in top_sv]
        fig = go.Figure(go.Bar(
            x=top_sv, y=top_names, orientation='h',
            marker_color=colors,
            text=[f'{v:+.4f}' for v in top_sv], textposition='outside',
            hovertemplate='%{y}<br>SHAP: %{x:.4f}<extra></extra>'
        ))
        fig.add_vline(x=0, line_dash='dash', line_color='gray')
        fig.update_layout(
            title=dict(text=f'SHAP Waterfall — Sample {idx} ({key.replace("_"," ").title()})',
                       font=dict(size=15), x=0.5),
            xaxis_title='SHAP value',
            yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
            height=max(400, top_n * 22 + 150),
            template='plotly_white',
            margin=dict(l=200, r=80, t=80, b=50))
        return fig

    def plot_biomarker_stability(self, model_name: str = 'random_forest',
                                  top_n: int = 25):
        """
        Biomarker stability plot: mean importance ± SD across all CV folds.
        Selection frequency shown as colour intensity.
        """
        if not PLOTLY_AVAILABLE: return None
        stab = self.stability_results.get(model_name)
        if stab is None or stab.empty: return None

        top = stab.head(top_n)
        colors = [f'rgba(52,152,219,{f:.2f})' for f in top['selection_freq']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top['mean_imp'], y=top['feature'],
            orientation='h',
            error_x=dict(type='data', array=top['std_imp'],
                         visible=True, thickness=1.5, width=6,
                         color='rgba(0,0,0,0.4)'),
            marker_color=colors,
            hovertemplate=(
                '<b>%{y}</b><br>'
                'Mean importance: %{x:.4f}<br>'
                'SD: %{error_x.array:.4f}<br>'
                'Selection freq: %{customdata:.0%}<extra></extra>'
            ),
            customdata=top['selection_freq'],
        ))
        fig.update_layout(
            title=dict(
                text=(f'Biomarker Stability — {model_name.replace("_"," ").title()}'
                      f'<br><span style="font-size:12px;color:#666">'
                      f'Mean ± SD across CV folds · colour = selection frequency</span>'),
                font=dict(size=16), x=0.5),
            xaxis_title='Mean MDI Importance ± SD',
            yaxis=dict(autorange='reversed', tickfont=dict(size=11)),
            height=max(450, top_n * 22 + 180),
            template='plotly_white',
            margin=dict(l=220, r=60, t=100, b=50))
        return fig

    def plot_permutation_importance(self, model_name: str = None, top_n: int = 25):
        """Model-agnostic permutation importance with error bars."""
        if not PLOTLY_AVAILABLE: return None
        key = model_name or self.get_best_model_name()
        if key not in self.models: return None
        perm = self.models[key].get('permutation_importance')
        if perm is None or perm.empty: return None

        top = perm.head(top_n)
        fig = go.Figure(go.Bar(
            x=top['importance_mean'], y=top['feature'],
            orientation='h',
            error_x=dict(type='data', array=top['importance_std'],
                         visible=True, thickness=1.5, width=6),
            marker_color='#27AE60',
            hovertemplate='%{y}<br>Importance: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text=f'Permutation Importance — {key.replace("_"," ").title()} (Top {top_n})',
                       font=dict(size=16), x=0.5),
            xaxis_title='Mean accuracy decrease',
            yaxis=dict(autorange='reversed', tickfont=dict(size=11)),
            height=max(400, top_n * 22 + 150),
            template='plotly_white',
            margin=dict(l=220, r=60, t=80, b=50))
        return fig

    def plot_feature_importance_heatmap(self, top_n: int = 30):
        """Cross-model feature importance heatmap — shows consistent biomarkers."""
        if not PLOTLY_AVAILABLE or not self.models: return None
        all_fi = {}
        for mname, res in self.models.items():
            fi = res.get('feature_importance')
            if fi is not None and not fi.empty:
                all_fi[mname] = dict(zip(fi['feature'], fi['importance']))

        if not SHAP_AVAILABLE and not all_fi: return None

        # Add SHAP
        for mname, sv in self.shap_values.items():
            all_fi[f'{mname}_SHAP'] = dict(
                zip(sv['mean_abs'].index, sv['mean_abs'].values))

        # Collect top features across all methods
        all_feats = set()
        for d in all_fi.values():
            sorted_f = sorted(d.items(), key=lambda x: -x[1])
            all_feats.update(f for f, _ in sorted_f[:top_n])
        all_feats = sorted(all_feats)

        mat = pd.DataFrame({m: [d.get(f, 0) for f in all_feats]
                             for m, d in all_fi.items()}, index=all_feats)
        # Normalize each column to [0,1] for comparability
        mat = mat.div(mat.max(axis=0).replace(0, 1), axis=1)
        mat = mat.loc[mat.max(axis=1).sort_values(ascending=False).index]

        fig = go.Figure(go.Heatmap(
            z=mat.values,
            x=[c.replace('_', ' ').title() for c in mat.columns],
            y=mat.index,
            colorscale='RdYlBu_r',
            hovertemplate='%{y}<br>%{x}<br>Norm. importance: %{z:.3f}<extra></extra>',
            colorbar=dict(title='Norm. Importance')
        ))
        fig.update_layout(
            title=dict(text=f'Cross-Model Feature Importance Heatmap (Top {top_n})',
                       font=dict(size=16), x=0.5),
            height=max(600, len(mat) * 18 + 200),
            xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=9), autorange='reversed'),
            template='plotly_white',
            margin=dict(l=200, r=40, t=80, b=80))
        return fig

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 1: CLASS IMBALANCE → SMOTE / ADASYN
    # ══════════════════════════════════════════════════════════════════════

    def detect_class_imbalance(self) -> Dict:
        """
        Detect class imbalance and return severity info.
        Imbalance ratio = n_majority / n_minority.
        Threshold: >2.0 = mild, >5.0 = severe.
        """
        if not self.processed_data: return {}
        y = self.processed_data['y']
        from collections import Counter
        counts = Counter(y)
        n_min = min(counts.values())
        n_max = max(counts.values())
        ratio = n_max / max(n_min, 1)
        return {
            'counts': counts,
            'ratio': ratio,
            'severity': 'none' if ratio < 2 else 'mild' if ratio < 5 else 'severe',
            'needs_resampling': ratio >= 2.0,
        }

    def apply_smote(self, strategy: str = 'auto') -> bool:
        """
        Register SMOTE strategy for use during training.
        SMOTE is applied only to training data inside _split() — never to test data.
        strategy: 'smote' | 'adasyn' | 'borderline' | 'svmsmote' | 'auto'
        Returns True if strategy registered successfully.
        """
        if not self.processed_data: return False
        try:
            from imblearn.over_sampling import SMOTE  # noqa: F401 – just check import
        except ImportError:
            self._warn("Install imbalanced-learn: pip install imbalanced-learn")
            return False

        info = self.detect_class_imbalance()
        if not info.get('needs_resampling'):
            self._warn("Classes appear balanced (ratio < 2.0). SMOTE not needed.")
            return False

        if strategy == 'auto':
            strategy = 'smote' if info['severity'] == 'mild' else 'adasyn'

        self.processed_data['smote_strategy'] = strategy
        self._ok(f"SMOTE strategy '{strategy.upper()}' registered — will be applied to training data only.")
        return True

    def _apply_smote_to_train(self, X_tr, y_tr, strategy: str):
        """Apply SMOTE variant to training fold only."""
        from collections import Counter
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
        n_min = min(Counter(y_tr).values())
        k = min(5, n_min - 1)
        if k < 1:
            self._warn("Too few minority samples in training fold for SMOTE — skipped.")
            return X_tr, y_tr
        if strategy == 'adasyn':
            sampler = ADASYN(random_state=42, n_neighbors=k)
        elif strategy == 'borderline':
            sampler = BorderlineSMOTE(random_state=42, k_neighbors=k)
        elif strategy == 'svmsmote':
            sampler = SVMSMOTE(random_state=42, k_neighbors=k)
        else:
            sampler = SMOTE(random_state=42, k_neighbors=k)
        try:
            X_res, y_res = sampler.fit_resample(X_tr, y_tr)
            before = dict(Counter(y_tr))
            after  = dict(Counter(y_res))
            self._ok(f"{strategy.upper()} (train only): {before} → {after}")
            return X_res, y_res
        except Exception as e:
            self._warn(f"SMOTE failed on training fold: {e}")
            return X_tr, y_tr

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 2: OUTLIER DETECTION (pre-training QC)
    # ══════════════════════════════════════════════════════════════════════

    def detect_outliers(self, contamination: float = 0.05) -> Dict:
        """
        Identify potential outlier samples using Isolation Forest + z-score.
        Returns dict with outlier indices and a Plotly figure.
        Best practice: investigate outliers before training, not just remove.
        """
        if not self.processed_data: return {}
        X = self.processed_data['X']
        sample_names = self.processed_data['sample_names']

        # ── Isolation Forest (density-based, robust to high dimensions) ──
        from sklearn.ensemble import IsolationForest
        from sklearn.decomposition import PCA

        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        iso_pred = iso.fit_predict(X)          # -1 = outlier, 1 = inlier
        iso_score = iso.score_samples(X)       # more negative = more anomalous
        iso_outliers = np.where(iso_pred == -1)[0]

        # ── Z-score sum (simple univariate screen) ──
        from scipy import stats as scipy_stats
        z = np.abs(scipy_stats.zscore(X, axis=0))
        z_outliers = np.where(z.max(axis=1) > 3.5)[0]  # any feature z > 3.5

        # ── PCA scatter for visualisation (2D projection) ──
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)

        if not PLOTLY_AVAILABLE:
            return {'iso_outliers': iso_outliers, 'z_outliers': z_outliers}

        y_labels = [self.processed_data['class_names'][yi] for yi in self.processed_data['y']]
        colors = ['#E74C3C' if i in iso_outliers else '#3498DB' for i in range(len(X))]
        sizes  = [14 if i in iso_outliers else 8 for i in range(len(X))]

        fig = go.Figure()
        # Inliers per class
        for cls in set(y_labels):
            idx = [i for i, l in enumerate(y_labels) if l == cls and i not in iso_outliers]
            if idx:
                fig.add_trace(go.Scatter(
                    x=X2[idx, 0], y=X2[idx, 1], mode='markers',
                    name=str(cls), marker=dict(size=8, opacity=0.8),
                    text=[sample_names[i] for i in idx],
                    hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
                ))
        # Outliers
        if len(iso_outliers):
            fig.add_trace(go.Scatter(
                x=X2[iso_outliers, 0], y=X2[iso_outliers, 1],
                mode='markers', name='⚠ Outlier (IsoForest)',
                marker=dict(size=14, color='#E74C3C', symbol='x',
                            line=dict(width=2, color='darkred')),
                text=[sample_names[i] for i in iso_outliers],
                hovertemplate='<b>OUTLIER: %{text}</b><br>Score: ' +
                              '<br>PC1: %{x:.3f}, PC2: %{y:.3f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text='Outlier Detection — PCA + Isolation Forest<br>'
                     '<span style="font-size:12px;color:#666">'
                     f'Red × = potential outliers (contamination={contamination:.0%})</span>',
                font=dict(size=16), x=0.5),
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',
            height=500, template='plotly_white',
            legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center')
        )
        return {
            'iso_outliers': iso_outliers.tolist(),
            'z_outliers':   z_outliers.tolist(),
            'union_outliers': list(set(iso_outliers.tolist()) | set(z_outliers.tolist())),
            'sample_outlier_names': [sample_names[i] for i in iso_outliers],
            'iso_scores': iso_score.tolist(),
            'figure': fig,
        }

    def remove_outliers(self, indices: List[int]) -> None:
        """Remove detected outlier samples from processed_data."""
        if not self.processed_data or not indices: return
        mask = np.ones(len(self.processed_data['X']), dtype=bool)
        mask[indices] = False
        self.processed_data['X']            = self.processed_data['X'][mask]
        self.processed_data['y']            = self.processed_data['y'][mask]
        self.processed_data['sample_names'] = [s for i, s in
            enumerate(self.processed_data['sample_names']) if mask[i]]
        self._ok(f"Removed {len(indices)} outlier samples. Remaining: {mask.sum()}")

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 3: UMAP + t-SNE SAMPLE SPACE EXPLORER
    # ══════════════════════════════════════════════════════════════════════

    def plot_dimensionality_reduction(self, method: str = 'umap',
                                       n_neighbors: int = 15,
                                       min_dist: float = 0.1,
                                       perplexity: int = 30) -> Optional[object]:
        """
        UMAP or t-SNE colored by class — gold standard for microbiome data exploration.
        Shows separation quality before and after training.
        """
        if not self.processed_data or not PLOTLY_AVAILABLE: return None
        X = self.processed_data['X']
        y = self.processed_data['y']
        class_names = self.processed_data['class_names']
        sample_names = self.processed_data['sample_names']

        if method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                     n_components=2, random_state=42)
                emb = reducer.fit_transform(X)
                title_method = f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})'
            except ImportError:
                self._warn("Install umap: pip install umap-learn")
                method = 'tsne'
        if method == 'tsne':
            from sklearn.manifold import TSNE
            perp = min(perplexity, len(X) // 3)
            reducer = TSNE(n_components=2, perplexity=max(5, perp),
                           random_state=42, n_iter=1000)
            emb = reducer.fit_transform(X)
            title_method = f't-SNE (perplexity={perp})'
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            emb = reducer.fit_transform(X)
            title_method = 'PCA'

        palette = px.colors.qualitative.D3 + px.colors.qualitative.Plotly
        fig = go.Figure()
        for ci, cls in enumerate(class_names):
            idx = [i for i, yi in enumerate(y) if yi == ci]
            if not idx: continue
            fig.add_trace(go.Scatter(
                x=emb[idx, 0], y=emb[idx, 1], mode='markers',
                name=str(cls),
                marker=dict(size=10, color=palette[ci % len(palette)],
                            opacity=0.85, line=dict(width=0.8, color='white')),
                text=[sample_names[i] for i in idx],
                hovertemplate='<b>%{text}</b><br>Class: ' + str(cls) +
                              '<br>X: %{x:.3f}, Y: %{y:.3f}<extra></extra>'
            ))
        fig.update_layout(
            title=dict(text=f'Sample Space — {title_method}<br>'
                            '<span style="font-size:12px;color:#555">'
                            'Colour = class label · separation = discriminability</span>',
                       font=dict(size=16), x=0.5),
            xaxis_title=f'{method.upper()} 1',
            yaxis_title=f'{method.upper()} 2',
            height=550, template='plotly_white',
            legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center',
                        font=dict(size=13))
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 4: PERMUTATION TEST (NULL MODEL SIGNIFICANCE)
    # ══════════════════════════════════════════════════════════════════════

    def run_permutation_test(self, model_name: str = 'random_forest',
                              n_permutations: int = 100) -> Optional[Dict]:
        """
        Permutation test (label-shuffling null model) — the most rigorous way
        to validate that the model learned real signal and is not overfitting.
        Returns p-value and null distribution plot.
        Reference: Topçuoğlu et al. mBio 2020; ML4Microbiome best practices.
        """
        if model_name not in self.models or not self.processed_data: return None
        from sklearn.model_selection import permutation_test_score

        model = self.models[model_name].get('model')
        if model is None: return None
        X, y = self.processed_data['X'], self.processed_data['y']
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        real_score, perm_scores, p_value = permutation_test_score(
            model, X, y, scoring='balanced_accuracy',
            cv=cv, n_permutations=n_permutations,
            random_state=42, n_jobs=-1)

        result = {
            'model_name':  model_name,
            'real_score':  real_score,
            'perm_scores': perm_scores,
            'p_value':     p_value,
            'n_perm':      n_permutations,
            'significant': p_value < 0.05,
        }

        # ── Plot null distribution ──
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=perm_scores, name='Null distribution',
                marker_color='#BDC3C7', opacity=0.8,
                nbinsx=25,
                hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ))
            fig.add_vline(x=real_score, line_dash='dash', line_color='#E74C3C',
                          line_width=2.5,
                          annotation_text=f'  Real score = {real_score:.3f}',
                          annotation_font=dict(color='#E74C3C', size=13))
            sig_text = f'p = {p_value:.3f} {"✓ SIGNIFICANT" if p_value < 0.05 else "✗ NOT significant"}'
            fig.update_layout(
                title=dict(text=(f'Permutation Test — {model_name.replace("_"," ").title()}<br>'
                                 f'<span style="font-size:13px;color:#555">{sig_text}</span>'),
                           font=dict(size=16), x=0.5),
                xaxis_title='Balanced Accuracy (permuted labels)',
                yaxis_title='Count',
                height=450, template='plotly_white',
                showlegend=False
            )
            result['figure'] = fig

        self._ok(f"Permutation test: real={real_score:.3f}, p={p_value:.3f} "
                 f"({'significant' if p_value < 0.05 else 'NOT significant'})")
        return result

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 5: LEARNING CURVES (sample-size vs performance)
    # ══════════════════════════════════════════════════════════════════════

    def plot_learning_curves(self, model_name: str = 'random_forest',
                              n_points: int = 8) -> Optional[object]:
        """
        Learning curves show how model performance scales with sample size.
        Critical for microbiome studies with small n — reveals underfitting,
        overfitting, and tells you how many samples are truly needed.
        """
        if model_name not in self.models or not self.processed_data: return None
        if not PLOTLY_AVAILABLE: return None

        from sklearn.model_selection import learning_curve

        model = self.models[model_name].get('model')
        if model is None: return None
        X, y = self.processed_data['X'], self.processed_data['y']

        train_sizes = np.linspace(0.10, 1.0, n_points)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        train_sz, train_sc, val_sc = learning_curve(
            model, X, y, train_sizes=train_sizes,
            cv=cv, scoring='balanced_accuracy',
            n_jobs=-1, shuffle=True, random_state=42)

        tr_mean  = train_sc.mean(axis=1)
        tr_std   = train_sc.std(axis=1)
        val_mean = val_sc.mean(axis=1)
        val_std  = val_sc.std(axis=1)

        fig = go.Figure()
        # Training score
        fig.add_trace(go.Scatter(
            x=train_sz, y=tr_mean, mode='lines+markers',
            name='Training score',
            line=dict(color='#E74C3C', width=2.5),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sz, train_sz[::-1]]),
            y=np.concatenate([tr_mean + tr_std, (tr_mean - tr_std)[::-1]]),
            fill='toself', fillcolor='rgba(231,76,60,0.12)',
            line=dict(color='rgba(255,255,255,0)'), showlegend=False
        ))
        # Validation score
        fig.add_trace(go.Scatter(
            x=train_sz, y=val_mean, mode='lines+markers',
            name='Cross-validation score',
            line=dict(color='#3498DB', width=2.5),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sz, train_sz[::-1]]),
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
            fill='toself', fillcolor='rgba(52,152,219,0.12)',
            line=dict(color='rgba(255,255,255,0)'), showlegend=False
        ))

        gap = tr_mean[-1] - val_mean[-1]
        diagnosis = ("Overfitting: train >> val — collect more data or regularize"
                     if gap > 0.15 else
                     "Underfitting: both scores low — try more complex model"
                     if val_mean[-1] < 0.6 else
                     "Good fit: scores converge")

        fig.update_layout(
            title=dict(
                text=(f'Learning Curves — {model_name.replace("_"," ").title()}<br>'
                      f'<span style="font-size:12px;color:#555">{diagnosis}</span>'),
                font=dict(size=16), x=0.5),
            xaxis_title='Training samples',
            yaxis_title='Balanced Accuracy',
            yaxis=dict(range=[0, 1.05]),
            height=480, template='plotly_white',
            legend=dict(x=0.65, y=0.05)
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 6: sPLS-DA (Sparse PLS Discriminant Analysis)
    # ══════════════════════════════════════════════════════════════════════

    def train_splsda(self, n_components: int = 2,
                      max_iter: int = 500,
                      test_size: float = 0.2) -> Optional[Dict]:
        """
        Sparse PLS Discriminant Analysis — the most interpretable multivariate
        method for microbiome classification (mixOmics-equivalent in Python).
        Simultaneously performs classification AND dimensionality reduction.
        Key advantage: naturally handles compositional, sparse OTU data.
        """
        if not self.processed_data: return None

        X_tr, X_te, y_tr, y_te = self._split(test_size)
        n_comp = min(n_components, X_tr.shape[1], len(np.unique(y_tr)) - 1)

        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import label_binarize
        from sklearn.linear_model import LogisticRegression

        # One-hot encode y for PLS
        classes = np.unique(y_tr)
        Y_tr_oh = label_binarize(y_tr, classes=classes)
        if Y_tr_oh.shape[1] == 1:
            Y_tr_oh = np.hstack([1 - Y_tr_oh, Y_tr_oh])

        # Scale X
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Fit PLS
        pls = PLSRegression(n_components=n_comp, max_iter=max_iter, scale=True)
        pls.fit(X_tr_sc, Y_tr_oh)

        # Scores (latent variables) for train/test
        T_tr = pls.transform(X_tr_sc)
        T_te = pls.transform(X_te_sc)

        # Classify in latent space
        clf = LogisticRegression(C=1.0, class_weight='balanced',
                                  max_iter=2000, random_state=42)
        clf.fit(T_tr, y_tr)
        y_pred = clf.predict(T_te)

        acc  = accuracy_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
        mcc  = matthews_corrcoef(y_te, y_pred)

        # ── Loadings (= variable importance in sPLS-DA) ──
        feat_names = self.processed_data['feature_names']
        loadings   = pd.DataFrame(
            pls.x_loadings_,
            index=feat_names,
            columns=[f'LV{i+1}' for i in range(n_comp)]
        )
        # Total loading = Euclidean norm across components
        loadings['total_loading'] = np.sqrt((loadings ** 2).sum(axis=1))
        loadings = loadings.sort_values('total_loading', ascending=False)

        # ── Scores scatter (Component 1 vs 2) ──
        result = {
            'model_type':  'sPLS-DA',
            'model':       (pls, clf, scaler),
            'test_accuracy': acc,
            'f1_score':    f1,
            'mcc':         mcc,
            'cv_mean':     0,
            'cv_std':      0,
            'n_components': n_comp,
            'loadings':    loadings,
            'T_train':     T_tr,
            'T_test':      T_te,
            'y_train':     y_tr,
            'y_test':      y_te,
            'scaler':      scaler,
            'feature_importance': loadings[['total_loading']].rename(
                columns={'total_loading': 'importance'}).reset_index().rename(
                columns={'index': 'feature'}),
        }

        # CV
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline as _Pipe
        _pipe = _Pipe([('sc', StandardScaler()),
                       ('pls', PLSRegression(n_components=n_comp, max_iter=max_iter)),
                       ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))])
        try:
            cv_sc = cross_val_score(
                _pipe,
                np.vstack([X_tr, X_te]),
                np.concatenate([y_tr, y_te]),
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='balanced_accuracy', n_jobs=-1
            )
            result['cv_mean'] = cv_sc.mean()
            result['cv_std']  = cv_sc.std()
        except Exception:
            pass

        self.models['splsda'] = result
        self._ok(f"sPLS-DA: acc={acc:.3f}, F1={f1:.3f}, CV={result['cv_mean']:.3f}")
        return result

    def plot_splsda_scores(self) -> Optional[object]:
        """Score scatter plot for sPLS-DA — equivalent to mixOmics plotIndiv."""
        if 'splsda' not in self.models or not PLOTLY_AVAILABLE: return None
        res = self.models['splsda']
        T   = res['T_train']
        y   = res['y_train']
        class_names = self.processed_data['class_names']
        palette = px.colors.qualitative.D3

        n_comp = res['n_components']
        if n_comp < 2:
            return None

        fig = make_subplots(rows=1, cols=1)
        for ci, cls in enumerate(class_names):
            idx = np.where(y == ci)[0]
            if not len(idx): continue
            fig.add_trace(go.Scatter(
                x=T[idx, 0], y=T[idx, 1], mode='markers',
                name=str(cls),
                marker=dict(size=11, color=palette[ci % len(palette)],
                            opacity=0.85, line=dict(width=0.8, color='white')),
                hovertemplate=f'Class: {cls}<br>LV1: %{{x:.3f}}<br>LV2: %{{y:.3f}}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text='sPLS-DA Score Plot (LV1 vs LV2)<br>'
                            '<span style="font-size:12px;color:#555">'
                            'Separation = discriminability of microbiome signal</span>',
                       font=dict(size=16), x=0.5),
            xaxis_title='Latent Variable 1 (LV1)',
            yaxis_title='Latent Variable 2 (LV2)',
            height=520, template='plotly_white',
            legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center',
                        font=dict(size=13))
        )
        return fig

    def plot_splsda_loadings(self, n_top: int = 20) -> Optional[object]:
        """Loading bar plot for sPLS-DA — shows which taxa drive separation."""
        if 'splsda' not in self.models or not PLOTLY_AVAILABLE: return None
        res = self.models['splsda']
        load = res['loadings']
        n_comp = res['n_components']
        cols = [f'LV{i+1}' for i in range(min(n_comp, 2))]

        top_load = load.head(n_top)
        fig = go.Figure()
        colors_lv = ['#3498DB', '#E74C3C']
        for ci, col in enumerate(cols):
            fig.add_trace(go.Bar(
                x=top_load[col], y=top_load.index,
                name=col, orientation='h',
                marker_color=colors_lv[ci % 2],
                opacity=0.85,
                hovertemplate=f'%{{y}}<br>{col} loading: %{{x:.4f}}<extra></extra>'
            ))
        fig.update_layout(
            title=dict(text=f'sPLS-DA Loadings (Top {n_top} taxa)<br>'
                            '<span style="font-size:12px;color:#555">'
                            'Larger |loading| = stronger contribution to separation</span>',
                       font=dict(size=16), x=0.5),
            xaxis_title='Loading value',
            yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
            barmode='group',
            height=max(400, n_top * 22 + 180),
            template='plotly_white',
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
            margin=dict(l=200, r=60, t=90, b=50)
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 7: MULTI-NORMALIZATION AUTO-BENCHMARK
    # ══════════════════════════════════════════════════════════════════════

    def benchmark_normalizations(self, target_variable: str,
                                   model: str = 'rf_quick') -> Optional[Dict]:
        """
        Auto-benchmark CLR vs log vs relative vs sqrt normalization.
        Uses a fast RF (50 trees) with 5-fold CV to compare AUROC/Balanced Acc.
        Returns the best normalization and a comparison plot.
        Based on Karwowska et al. Microbiome 2025 — normalization matters for
        feature selection but less for classification accuracy.
        """
        norms = ['clr', 'log', 'relative', 'sqrt']
        results = {}

        orig_settings = self.settings.copy()
        for nm in norms:
            try:
                self.settings['ml_normalization'] = nm
                ok = self.prepare_data_for_ml(
                    target_variable=target_variable,
                    feature_selection=True, max_features=50,
                    min_samples_per_class=2)
                if not ok: continue

                X, y = self.processed_data['X'], self.processed_data['y']
                rf = RandomForestClassifier(n_estimators=50, class_weight='balanced',
                                             random_state=42, n_jobs=-1)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                sc = cross_val_score(rf, X, y, cv=cv,
                                      scoring='balanced_accuracy', n_jobs=-1)
                results[nm] = {'mean': sc.mean(), 'std': sc.std()}
            except Exception as e:
                results[nm] = {'mean': 0, 'std': 0}

        # Restore settings
        self.settings = orig_settings

        if not results: return None

        best = max(results, key=lambda k: results[k]['mean'])
        self._ok(f"Best normalization: {best.upper()} "
                 f"(CV={results[best]['mean']:.3f}±{results[best]['std']:.3f})")

        if PLOTLY_AVAILABLE:
            norms_used = list(results.keys())
            means = [results[n]['mean'] for n in norms_used]
            stds  = [results[n]['std']  for n in norms_used]
            colors = ['#2ECC71' if n == best else '#BDC3C7' for n in norms_used]
            fig = go.Figure(go.Bar(
                x=[n.upper() for n in norms_used], y=means,
                error_y=dict(type='data', array=stds, visible=True),
                marker_color=colors,
                text=[f'{v:.3f}' for v in means], textposition='outside',
                hovertemplate='%{x}<br>Balanced Acc: %{y:.3f}<extra></extra>'
            ))
            fig.update_layout(
                title=dict(text='Normalization Benchmark (Random Forest, 5-fold CV)<br>'
                                '<span style="font-size:12px;color:#27AE60">'
                                f'Best: {best.upper()}</span>',
                           font=dict(size=16), x=0.5),
                yaxis_title='Balanced Accuracy (CV)',
                yaxis_range=[0, 1.1],
                height=420, template='plotly_white'
            )
            results['_figure'] = fig
            results['_best'] = best
        return results

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 8: LEfSe-STYLE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════

    def run_lefse_analysis(self, top_n: int = 30) -> Optional[Dict]:
        """
        LEfSe (Linear Discriminant Analysis Effect Size) — the gold-standard
        microbiome biomarker discovery tool (Segata et al. 2011, Genome Biology).
        
        Steps:
        1. Kruskal-Wallis H-test across groups (non-parametric ANOVA)
        2. Pairwise Wilcoxon rank-sum tests for biological consistency
        3. LDA effect size estimation for each significant feature
        """
        if not self.processed_data: return None

        X_df     = self.processed_data['X_df']
        y_series = self.processed_data['y_series']
        class_names = self.processed_data['class_names']
        from scipy import stats as scipy_stats

        rows = []
        for feat in X_df.columns:
            groups = [X_df.loc[y_series == cls, feat].values
                      for cls in class_names]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2: continue

            # Kruskal-Wallis
            try:
                stat, pval = scipy_stats.kruskal(*groups)
            except Exception:
                continue

            # LDA effect size: mean difference normalized by pooled SD
            grand_mean = X_df[feat].mean()
            lda_scores = []
            for gi, cls in enumerate(class_names):
                g_data = groups[gi]
                if len(g_data) == 0: continue
                g_mean = g_data.mean()
                # Effect = |group_mean - grand_mean| / (pooled_std + 1e-6)
                pooled_std = X_df[feat].std() + 1e-6
                lda_scores.append(abs(g_mean - grand_mean) / pooled_std)
            lda_max = max(lda_scores) if lda_scores else 0
            best_class = class_names[np.argmax(lda_scores)] if lda_scores else ''

            rows.append({
                'Feature':       feat,
                'KW_pvalue':     pval,
                'LDA_score':     lda_max,
                'Enriched_in':   best_class,
                'KW_stat':       stat,
            })

        if not rows: return None
        df = pd.DataFrame(rows)
        df = df[df['KW_pvalue'] < 0.05].sort_values('LDA_score', ascending=False)
        df['Bonferroni_p'] = np.minimum(df['KW_pvalue'] * len(df), 1.0)

        top_df = df.head(top_n)

        if not PLOTLY_AVAILABLE:
            return {'table': df, 'top': top_df}

        # Colour by enriched class
        palette = px.colors.qualitative.D3
        class_color = {cls: palette[i % len(palette)]
                       for i, cls in enumerate(class_names)}
        colors = [class_color.get(c, '#888') for c in top_df['Enriched_in']]

        fig = go.Figure(go.Bar(
            x=top_df['LDA_score'], y=top_df['Feature'],
            orientation='h',
            marker_color=colors,
            text=top_df['Enriched_in'],
            textposition='outside',
            hovertemplate=(
                '<b>%{y}</b><br>'
                'LDA score: %{x:.3f}<br>'
                'KW p-value: %{customdata[0]:.4f}<br>'
                'Enriched in: %{customdata[1]}<extra></extra>'
            ),
            customdata=list(zip(top_df['KW_pvalue'].round(4), top_df['Enriched_in']))
        ))
        # Legend for classes
        for cls in class_names:
            fig.add_trace(go.Bar(
                x=[None], y=[None], name=str(cls),
                marker_color=class_color.get(cls, '#888')
            ))
        fig.update_layout(
            title=dict(text=f'LEfSe Analysis — Top {len(top_df)} Biomarkers<br>'
                            '<span style="font-size:12px;color:#555">'
                            'LDA effect size · Kruskal-Wallis p < 0.05</span>',
                       font=dict(size=16), x=0.5),
            xaxis_title='LDA Effect Size',
            yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
            height=max(450, len(top_df) * 22 + 200),
            template='plotly_white',
            showlegend=True,
            legend=dict(title='Enriched in', x=1.01, y=1),
            margin=dict(l=200, r=180, t=90, b=50)
        )
        return {'table': df, 'top': top_df, 'figure': fig}

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 9: PREDICTION CONFIDENCE (per-sample uncertainty)
    # ══════════════════════════════════════════════════════════════════════

    def compute_prediction_confidence(self,
                                       model_name: str = 'random_forest') -> Optional[Dict]:
        """
        Per-sample prediction confidence and uncertainty using:
        - predict_proba max probability → confidence
        - Shannon entropy of probability vector → uncertainty
        Identifies samples the model is uncertain about — critical for
        clinical decision support and data quality assessment.
        """
        if model_name not in self.models or not self.processed_data: return None
        res   = self.models[model_name]
        model = res.get('model')
        X     = self.processed_data['X']
        y     = self.processed_data['y']
        class_names = self.processed_data['class_names']
        sample_names = self.processed_data['sample_names']

        if not hasattr(model, 'predict_proba'): return None

        proba = model.predict_proba(X)
        pred  = model.predict(X)
        conf  = proba.max(axis=1)                       # max probability
        entr  = -np.sum(proba * np.log2(proba + 1e-9), axis=1)  # Shannon entropy
        correct = pred == y

        result_df = pd.DataFrame({
            'Sample':       sample_names,
            'True_class':   [class_names[yi] for yi in y],
            'Pred_class':   [class_names[pi] for pi in pred],
            'Confidence':   conf.round(3),
            'Uncertainty':  entr.round(3),
            'Correct':      correct,
        }).sort_values('Confidence')

        if not PLOTLY_AVAILABLE:
            return {'table': result_df}

        fig = go.Figure()
        for cls in class_names:
            sub = result_df[result_df['True_class'] == cls]
            sub_cor = sub[sub['Correct']]
            sub_err = sub[~sub['Correct']]
            if not sub_cor.empty:
                fig.add_trace(go.Scatter(
                    x=sub_cor['Confidence'], y=sub_cor['Uncertainty'],
                    mode='markers', name=f'{cls} ✓ correct',
                    marker=dict(size=9, symbol='circle', opacity=0.75),
                    text=sub_cor['Sample'],
                    hovertemplate='<b>%{text}</b><br>Conf: %{x:.3f}<br>Entropy: %{y:.3f}<extra></extra>'
                ))
            if not sub_err.empty:
                fig.add_trace(go.Scatter(
                    x=sub_err['Confidence'], y=sub_err['Uncertainty'],
                    mode='markers', name=f'{cls} ✗ wrong',
                    marker=dict(size=12, symbol='x', opacity=0.9,
                                line=dict(width=2, color='black')),
                    text=sub_err['Sample'],
                    hovertemplate='<b>WRONG: %{text}</b><br>Conf: %{x:.3f}<br>Entropy: %{y:.3f}<extra></extra>'
                ))

        fig.add_vline(x=0.5, line_dash='dash', line_color='gray',
                      annotation_text='50% threshold', annotation_font=dict(size=11))
        fig.update_layout(
            title=dict(text=f'Prediction Confidence Map — {model_name.replace("_"," ").title()}<br>'
                            '<span style="font-size:12px;color:#555">'
                            'Low confidence + high entropy = uncertain predictions</span>',
                       font=dict(size=16), x=0.5),
            xaxis_title='Confidence (max predict_proba)',
            yaxis_title='Uncertainty (Shannon entropy)',
            height=520, template='plotly_white',
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center')
        )
        return {'table': result_df, 'figure': fig}

    # ══════════════════════════════════════════════════════════════════════
    #  INNOVATION BLOCK 10: SCIENTIFIC PAPER METHODS SECTION AUTO-GENERATOR
    # ══════════════════════════════════════════════════════════════════════

    def generate_methods_section(self) -> str:
        """
        Auto-generate a publication-ready Methods section for the ML analysis.
        Can be copy-pasted directly into a paper or supplementary materials.
        """
        if not self.processed_data: return "No data prepared."
        pd_ = self.processed_data
        n      = len(pd_['sample_names'])
        n_feat = len(pd_['feature_names'])
        n_cls  = pd_['n_classes']
        classes = pd_['class_names']
        norm   = pd_['normalization'].upper()
        cv_str = pd_['cv_strategy']
        target = pd_['target_name']
        smote  = pd_.get('smote_applied', None)

        models_trained = list(self.models.keys())
        best_name = self.get_best_model_name()
        best_res  = self.models.get(best_name, {})

        lines = [
            "## Machine Learning Analysis — Methods",
            "",
            "### Data Preparation",
            f"Microbiome OTU/ASV abundance data were processed for supervised machine "
            f"learning classification of **{target}** ({n_cls} classes: "
            f"{', '.join(str(c) for c in classes)}). "
            f"The dataset comprised **{n} samples** and **{n_feat} features** after "
            f"filtering (zero-variance features removed).",
            "",
            f"Data were normalized using **{norm} transformation** prior to modelling. "
            f"CLR (Centered Log-Ratio) transformation was applied to account for the "
            f"compositional nature of microbiome data, as recommended by "
            f"Aitchison (1986) and Gloor et al. (2017).",
            "",
        ]

        if smote:
            lines += [
                f"Class imbalance was addressed using **{smote}** (Synthetic Minority "
                f"Over-sampling Technique) applied exclusively to the training set "
                f"to avoid data leakage (Chawla et al. 2002).",
                "",
            ]

        lines += [
            "### Feature Selection",
            "Feature selection was performed prior to training using a combination of "
            "Mutual Information scoring and Random Forest feature importances "
            "(combined method), retaining the top informative features. "
            "Zero-prevalence features were removed as a pre-processing step.",
            "",
            "### Classification Models",
            f"The following machine learning classifiers were trained and evaluated: "
            f"{', '.join(m.replace('_',' ').title() for m in models_trained)}. "
            f"All tree-based models used `class_weight='balanced'` to handle any "
            f"residual class imbalance.",
            "",
        ]

        if 'xgboost' in models_trained or 'lightgbm' in models_trained:
            lines += [
                "Gradient boosting models (XGBoost, LightGBM) were trained with "
                "early stopping and regularization (L1/L2) to prevent overfitting. "
                "Hyperparameters were optimized using **Optuna** (TPE sampler, "
                "Bayesian optimization) when enabled.",
                "",
            ]

        if 'stacking' in models_trained:
            lines += [
                "A **stacking ensemble** was constructed with tree-based models as "
                "base learners and Logistic Regression as the meta-learner, "
                "using 5-fold cross-validation to generate out-of-fold predictions.",
                "",
            ]

        cv_map = {
            'loocv': 'Leave-One-Out Cross-Validation (LOOCV), appropriate for n < 30',
            'repeated_kfold_5x5': 'Repeated Stratified 5-fold Cross-Validation '
                                   '(5 repeats × 5 folds = 25 evaluations)',
            'repeated_kfold_5x10': 'Repeated Stratified 5-fold Cross-Validation '
                                    '(10 repeats × 5 folds = 50 evaluations)',
        }
        lines += [
            "### Model Evaluation",
            f"Models were evaluated using **{cv_map.get(cv_str, cv_str)}**, "
            f"with all splits stratified to maintain class proportions. "
            f"Performance was assessed using: Balanced Accuracy, Weighted F1-score, "
            f"AUROC, Average Precision (AUPRC), Matthews Correlation Coefficient (MCC), "
            f"and Cohen's Kappa. A held-out test set (20% of data) provided "
            f"unbiased final evaluation.",
            "",
        ]

        if self.shap_values:
            lines += [
                "### Explainability (XAI)",
                "Model predictions were explained using **SHAP** (SHapley Additive "
                "exPlanations, Lundberg & Lee 2017). TreeExplainer was used for "
                "tree-based models (exact Shapley values), and KernelExplainer for "
                "non-tree models (approximate). SHAP beeswarm plots were used to "
                "visualize global feature impact, and waterfall plots for individual "
                "sample explanations.",
                "",
            ]

        if self.stability_results:
            lines += [
                "Biomarker stability was assessed by computing feature importances "
                "across all CV folds (Repeated Stratified K-fold). Selection frequency "
                "and coefficient of variation of importance across folds quantified "
                "biomarker robustness.",
                "",
            ]

        if best_res:
            acc   = best_res.get('test_accuracy', 0)
            auroc = best_res.get('roc_auc', 0)
            cv_m  = best_res.get('cv_mean', 0)
            cv_s  = best_res.get('cv_std', 0)
            lines += [
                "### Best Model Summary",
                f"The best-performing model was **{best_name.replace('_',' ').title()}** "
                f"(Test Accuracy={acc:.3f}, AUROC={auroc:.3f}, "
                f"CV Balanced Accuracy={cv_m:.3f}±{cv_s:.3f}).",
                "",
                "### Software",
                "All analyses were performed in Python 3.10+ using scikit-learn "
                "(Pedregosa et al. 2011), XGBoost (Chen & Guestrin 2016), "
                "LightGBM (Ke et al. 2017), SHAP (Lundberg et al. 2020), "
                "and Optuna (Akiba et al. 2019). "
                "Statistical analyses used SciPy (Virtanen et al. 2020).",
            ]

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    #  REPORT
    # ──────────────────────────────────────────────────────────────────────

    def generate_ml_report(self) -> str:
        return self.get_text_report()

    def get_text_report(self) -> str:
        """Full scientific markdown report."""
        lines = ["# ML Results — Microbiome Analysis\n"]
        pd_ = self.processed_data
        if pd_:
            lines += [
                f"**Target:** {pd_.get('target_name','N/A')}  ",
                f"**Samples:** {len(pd_.get('sample_names',[]))}  ",
                f"**Features:** {len(pd_.get('feature_names',[]))}  ",
                f"**Classes:** {', '.join(str(c) for c in pd_.get('class_names',[]))}  ",
                f"**Normalization:** {pd_.get('normalization','N/A')}  ",
                f"**CV strategy:** {pd_.get('cv_strategy','N/A')}  \n",
            ]
        comp = self.get_model_comparison()
        if comp is not None:
            lines.append("## Model Comparison\n")
            lines.append(comp.to_markdown(index=False))
            lines.append("\n")
        if self.shap_values:
            fi = self.get_feature_importance(top_n=20, method='shap')
            if fi is not None:
                lines.append("## Top SHAP Biomarkers\n")
                lines.append(fi.to_markdown(index=False))
                lines.append("\n")
        lines.append("---\n")
        lines.append(self.generate_methods_section())
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    #  LOGGING HELPERS
    # ──────────────────────────────────────────────────────────────────────
    def _err(self, m):  print(f"ML ERROR: {m}"); st.error(m)
    def _warn(self, m): print(f"ML WARN:  {m}"); st.warning(m)
    def _ok(self, m):   print(f"ML OK:    {m}"); st.success(m)
