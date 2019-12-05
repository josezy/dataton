import eli5
import numpy as np
import pandas as pd
import seaborn as sns

from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier

from eli5.sklearn import PermutationImportance
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from dataton import balance_data, build_model

# LOAD DATA
LITE = True
fts_filename = 'features_train_lite.csv' if LITE else 'features_train.csv'

full_feats = pd.read_csv(f'data/features/{fts_filename}')
rpta = full_feats.var_rpta
features = full_feats.iloc[:, 1:-1]

# balance this shit
X, y = balance_data(features, rpta, max_rptas=200, balance_factor=1.0)


# Heatmap correlations
plt.figure()
data = X.join(y)
corrmat = data.corr()
rpta_corr = corrmat['var_rpta'][:]
y_pos = np.arange(len(data.columns))
plt.bar(y_pos, np.abs(rpta_corr), color=(0.6, 0.4, 0.3, 0.6))
plt.ylim(0, 0.4)
plt.xticks(y_pos, data.columns, rotation='vertical')
plt.xlabel('Feature name')
plt.ylabel('Correlation')
plt.title('Correlation Feature Analisys')
plt.tight_layout()
plt.grid()
plt.draw()


# Feature importance
plt.figure()
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20)[::-1].plot(kind='barh')
plt.show()


# Univariate Selection
# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(X, y)  # fix non-negative
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
# print(featureScores.nlargest(20, 'Score'))  # print 10 best features


# plot feature importance
model = XGBClassifier()
model.fit(X, y)
plot_importance(model)
plt.draw()

# cross-validation LENTOO
# svc = SVC(kernel="linear")
# rfecv = RFECV(estimator=svc)
# rfecv.fit(X, y)
# y_pos = np.arange(len(X.columns))
# plt.bar(y_pos, rfecv.grid_scores_, color=(0.2, 0.4, 0.6, 0.6))
# plt.ylim(0.0, 1.0)
# plt.xlabel('Number of features selected')
# plt.ylabel('Cross validation score (nb of correct classifications)')
# plt.title('Feature Analisys')
# plt.draw()


# Use feature importance
def base_model():
    return build_model(input_dim=len(X.columns))


# evaluate model with standardized dataset LENTOO
# estimator = KerasClassifier(build_fn=base_model, epochs=100, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, features, rpta, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


model = KerasClassifier(build_fn=base_model)
model.fit(X, y, epochs=50, batch_size=128)

perm = PermutationImportance(model, random_state=1).fit(X, y)
print(eli5.show_weights(perm, feature_names=X.columns.tolist()).data)

plt.show()
