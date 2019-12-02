import eli5
# import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt

# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV

from eli5.sklearn import PermutationImportance
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# from code import load_data
from code import balance_data, build_model

# LOAD DATA
# ids, data, rpta = load_data()
full_feats = pd.read_csv('data/features_lite_train.csv')
rpta = full_feats.var_rpta
features = full_feats.iloc[:, 1:-1]

# balance this shit
bal_data, bal_rpta = balance_data(
    features, rpta, max_rptas=200, balance_factor=1.0)


# plot feature importance
model = XGBClassifier()
model.fit(bal_data, bal_rpta)
plot_importance(model)
plt.draw()

# svc = SVC(kernel="linear")
# rfecv = RFECV(estimator=svc)
# rfecv.fit(bal_data, bal_rpta)

# y_pos = np.arange(len(bal_data.columns))
# plt.bar(y_pos, rfecv.grid_scores_, color=(0.2, 0.4, 0.6, 0.6))
# plt.ylim(0.0, 1.0)
# plt.xlabel('Number of features selected')
# plt.ylabel('Cross validation score (nb of correct classifications)')
# plt.title('Feature Analisys')
# plt.draw()


# Use feature importance
def base_model():
    return build_model(input_dim=len(features.columns))


model = KerasRegressor(build_fn=base_model)
model.fit(features, rpta, epochs=30, batch_size=128)

perm = PermutationImportance(model, random_state=1).fit(features, rpta)
print(eli5.show_weights(perm, feature_names=features.columns.tolist()).data)

plt.show()
