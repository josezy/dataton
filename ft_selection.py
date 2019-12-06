from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SelectKBest, f_classif


class PipelineRFE(Pipeline):
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self


pipe = PipelineRFE([
    ('std_scaler', StandardScaler()),
    ("ET", ExtraTreesRegressor(random_state=42, n_estimators=250))
])
rfecv_selector = RFECV(pipe, cv=10, step=2, scoring="neg_mean_squared_error", n_jobs=-1)

# Other
kbest_selector = SelectKBest(f_classif, k=20)
