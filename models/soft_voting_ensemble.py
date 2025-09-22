from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Backward-compatible soft voting ensemble.
    Resolves base estimators from many possible legacy attributes:
    - models / models_
    - member_models (dict of name->estimator)
    - estimators / estimators_ (optionally [(name, est), ...])
    - base_models / base_estimators / model_list / learners / clfs / classifiers
    - named_estimators_ / named_models_
    """

    def __init__(self, models=None, weights=None):
        self.models = models
        self.weights = weights

    # INTERNALS
    def _to_list(self, v):
        if v is None:
            return []
        if isinstance(v, dict):
            v = list(v.values())
        if isinstance(v, (list, tuple)):
            # flatten [(name, est), ...] -> [est, ...]
            if v and isinstance(v[0], (list, tuple)) and len(v[0]) == 2:
                try:
                    v = [e for _, e in v]
                except Exception:
                    pass
            return list(v)
        return [v]

    def _looks_like_estimator(self, e):
        return hasattr(e, "fit") and (hasattr(e, "predict_proba") or hasattr(e, "decision_function"))

    def _resolve_models(self):
        # 1) use canonical if present
        if getattr(self, "models", None):
            return [e for e in self._to_list(self.models) if self._looks_like_estimator(e)]

        # 2) try common/legacy names
        candidates = [
            "member_models",
            "models_", "estimators_", "estimators",
            "base_models", "base_estimators",
            "model_list", "learners", "clfs", "classifiers",
            "named_estimators_", "named_models_",
        ]
        for name in candidates:
            if hasattr(self, name):
                out = [e for e in self._to_list(getattr(self, name)) if self._looks_like_estimator(e)]
                if out:
                    # cache canonically for next time
                    self.models = list(out)
                    return self.models

        # 3) last resort: scan __dict__ for model-ish attrs
        for k, v in getattr(self, "__dict__", {}).items():
            if not any(tok in k.lower() for tok in ["member_models", "model", "estim", "clf", "learner"]):
                continue
            out = [e for e in self._to_list(v) if self._looks_like_estimator(e)]
            if out:
                self.models = list(out)
                return self.models

        return []

    # SKLEARN API
    def fit(self, X, y):
        ms = self._resolve_models()
        if not ms:
            raise AttributeError("SoftVotingEnsemble has no base models to fit.")
        fitted = []
        for est in ms:
            est.fit(X, y)
            fitted.append(est)
        self.models = fitted
        # classes_
        if hasattr(fitted[0], "classes_"):
            self.classes_ = getattr(fitted[0], "classes_")
        else:
            self.classes_ = np.unique(y)
        return self

    def _proba_from_est(self, est, X):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)
        # decision_function -> softmax fallback
        s = est.decision_function(X)
        if s.ndim == 1:
            s = np.vstack([-s, s]).T
        e = np.exp(s - np.max(s, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        ms = self._resolve_models()
        if not ms:
            raise AttributeError("SoftVotingEnsemble has no base models (tried models/member_models/estimators[_]/...).")

        # If any child is another SoftVotingEnsemble / VotingClassifier, let it resolve itself
        probs_list = []
        for est in ms:
            if type(est).__name__ in ("SoftVotingEnsemble", "VotingClassifier"):
                # ensure nested ensembles resolve their models when called
                pass
            p = self._proba_from_est(est, X)
            probs_list.append(p)

        probs = np.stack(probs_list, axis=0)  # [n_models, n_rows, n_classes]

        # weights
        if getattr(self, "weights", None) is None:
            w = np.ones(probs.shape[0], dtype=float) / probs.shape[0]
        else:
            w = np.asarray(self.weights, dtype=float)
            if w.shape[0] != probs.shape[0]:
                w = np.resize(w, probs.shape[0])
            s = w.sum()
            w = w / s if s != 0 else np.ones_like(w) / w.size

        return np.tensordot(w, probs, axes=(0, 0))

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx] if hasattr(self, "classes_") else idx

    # PICKLE RESILIENCE
    def __getstate__(self):
        state = self.__dict__.copy()
        # ensure 'models' key is present if resolvable
        if not state.get("models"):
            resolved = self._resolve_models()
            if resolved:
                state["models"] = list(resolved)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # nothing else required; _resolve_models will populate on demand
