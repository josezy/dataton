import time
import psycopg2

import xgboost as xgb
import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

from fts_utils import (
    fts_totales,
    fts_var_rpta,
    fts_sesiones,
    fts_modas,
    fts_ratio,
    fts_sesion_stats,
    fts_fecha_trxn,
    fts_ratios_maestro,
    fts_month_entropy,
    fts_horadia_entropy,
    fts_interval,
    fts_culpabanco_entropy,
    fts_umbrales,
)


PREDICT = True
_LITE = False
_CACHE = True
NO_SEGMENTO = False

_BALANCE_FACTOR = 1.0  # relation between number of 0 & 1
GRID_SEARCH = False

# Keras constants
_EPOCHS = 500
_BATCH_SIZE = 128
_LOAD_MODEL = False
VAL_ACC_TH = 0.8

PARAM = {
    'eta': 0.5,
    'max_depth': 1,
    'gamma': 0.0,
    'min_child_weight': 5,
    'colsample_bytree': 0.5,
    'objective': 'binary:logistic'
}
NUM_ROUND = 2

GRID_PARAMS = {
    "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    # "min_child_weight": [1, 3, 5, 7],
    # "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    # "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
}

# GRID_PARAMS = {
#     'batch_size': [512, 1024],
#     'nb_epoch': [50, 100, 200],
#     'optimizer': ['sgd', 'adam']
# }


con = psycopg2.connect(
    database="dataton",
    user="dataton",
    password="dataton",
    host="localhost",
    port="5433"
)


def select_features(data, rpta):
    # return [
    #     # 'total_trxn', 'total_sesiones',
    #     'segmento',
    #     # 'times_used_disp',
    #     # 'times_used_canal',
    #     # 'ratio_financiera',
    #     # 'ratio_exitosa',
    #     # 'ratio_no_exitosa',
    #     # 'total_atypical_sesion',
    #     # 'avg_normal_sesion',
    #     # 'total_office_trxn', 'total_night_trxn',
    #     # 'total_weekday_trxn', 'total_weekend_trxn',
    #     # 'high_season_trxn', 'low_season_trxn',
    #     # 'month_entropy',
    #     'trxn_horadia_entropy',
    #     # 'avg_trxn_interval',
    #     # 'trxn_week_entropy',
    #     # 'umbral_trxn_lun', 'umbral_trxn_mar',
    #     # 'umbral_trxn_mie', 'umbral_trxn_jue',
    #     # 'umbral_trxn_vie',
    #     # 'umbral_trxn_sab', 'umbral_trxn_dom'
    # ]  # Hardcoded features

    bal_data, bal_rpta = balance_data(data, rpta, max_ones=500)
    # Select by KBest
    kbest_selector = SelectKBest(f_classif, k=10)
    kbest_selector.fit_transform(bal_data, bal_rpta)
    f_score_indexes = (-kbest_selector.scores_).argsort()[:10]
    kbest_features = data.columns[f_score_indexes].to_list()

    # Select by correlation matrix
    data = bal_data.join(bal_rpta)
    corrmat = data.corr()
    rpta_corr = abs(corrmat['var_rpta'][:].drop('var_rpta'))
    top = rpta_corr[rpta_corr > rpta_corr.mean()]
    corr_features = top.index.to_list()

    return list(set([
        *kbest_features,
        # *corr_features,
    ]))


def load_data(training=True, lite=True, cache=True):
    lite = lite if training else False
    print(f"[+] Loading data: training={training}, lite={lite}, cache={cache}")
    train_str = '_train' if training else '_predict'
    lite_str = '_lite' if lite else ''
    feats_file_name = f"features{train_str}{lite_str}.csv"

    if cache:
        try:
            fts = pd.read_csv(f'data/features/{feats_file_name}')
            print("Using cached features from ", feats_file_name, len(fts.columns))
            print("\tFeatures:", fts.iloc[:, 1:-1].columns)
            if training:
                return fts.id, fts.iloc[:, 1:-1], fts.var_rpta
            else:
                return fts.id, fts.iloc[:, 1:], None
        except FileNotFoundError:
            print("Could not use cached features:", feats_file_name)

    data_table_name = (
        f'datos_transaccionales_train{lite_str}'
        if training
        else 'datos_transaccionales_predict'
    )
    rpta_table_name = (
        f'datos_var_rpta_train{lite_str}'
        if training
        else 'ids_predict'
    )
    start = time.time()

    print("umbrales")
    umbrales = fts_umbrales(con, data_table_name)

    print("culpabanco_entropy")
    culpabanco_entropy = fts_culpabanco_entropy(con, data_table_name)

    print("avg_trxn_interval")
    intervals = fts_interval(con, data_table_name)

    print("horadia_entropy")
    horadia_entropy = fts_horadia_entropy(con, data_table_name)

    print("month_entropy")
    month_entropy = fts_month_entropy(con, data_table_name)

    print("ratio_descripcion_grupo, ratio_producto_asociado")
    ratios_maestro = fts_ratios_maestro(con, data_table_name)

    print("total_office_trxn, total_night_trxn")
    fecha_trxns = fts_fecha_trxn(con, data_table_name)

    print("total_atypical_sesion, avg_normal_sesion")
    sesiones = fts_sesion_stats(con, data_table_name)

    print("ratio_financiera, ratio_exitosa")
    ratios = fts_ratio(con, data_table_name)

    print("moda_cdgtrn, moda_cdgrpta")
    modas = fts_modas(con, data_table_name)

    print("times_used_disp, times_used_canal")
    disposit, canal = fts_sesiones(con, data_table_name)

    print("total_trxn, total_sesiones")
    totales = fts_totales(con, data_table_name)

    print("year_analisis, segmento")
    var_rpta = fts_var_rpta(con, rpta_table_name, training)

    features = pd.concat([
        totales.total_trxn,
        totales.total_sesiones,
        var_rpta.year_analisis,
        var_rpta.segmento,
        disposit.times_used_disp,
        canal.times_used_canal,
    ], axis=1)
    features = features.join(modas, on='id')
    features = features.join(ratios, on='id')
    features = features.join(sesiones, on='id')
    features = features.join(fecha_trxns, on='id')
    features = features.join(ratios_maestro, on='id')
    features = features.join(month_entropy, on='id')
    features = features.join(horadia_entropy, on='id')
    features = features.join(intervals, on='id')
    features = features.join(culpabanco_entropy, on='id')
    features = features.join(umbrales, on='id')

    features = features.fillna(0)
    scaler = StandardScaler()
    features[features.columns] = scaler.fit_transform(features[features.columns])

    full_features = pd.concat([
        features,
        var_rpta.get('var_rpta')
    ], axis=1)
    full_features.to_csv(f'data/features/{feats_file_name}')
    print("[+] Feature extraction took:", time.time() - start, "seconds")
    print("\tIt calculated", len(features.columns), "features")
    return var_rpta.index.to_series(), features, var_rpta.get('var_rpta')


def balance_data(data, rpta, max_ones=None, balance_factor=1.0):
    # balance number of 0 and 1's
    max_ones = max_ones or int(rpta.value_counts().min() * balance_factor)
    bal_data = pd.concat([
        data[rpta == 0][:int(max_ones * balance_factor)],
        data[rpta == 1][:max_ones]
    ])
    bal_rpta = pd.concat([
        rpta[rpta == 0][:int(max_ones * balance_factor)],
        rpta[rpta == 1][:max_ones]
    ])
    return bal_data, bal_rpta


def split_train_test(data, rpta, tt_threshold=0.7):
    threshold = int(len(rpta) * tt_threshold)
    train_data = data.iloc[:threshold]
    test_data = data.iloc[threshold:]
    train_rpta = rpta.iloc[:threshold]
    test_rpta = rpta.iloc[threshold:]
    return train_data, train_rpta, test_data, test_rpta


def base_model(input_dim, optimizer='sgd'):
    # Model definition
    model = keras.Sequential([
        keras.layers.Dense(12, input_dim=input_dim, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['acc']
    )
    return model


if __name__ == '__main__':
    _, data, rpta = load_data(lite=_LITE, cache=_CACHE)

    print("[+] Selecting features...")
    selected_features = select_features(data, rpta)
    # Work without segmento
    if NO_SEGMENTO and 'segmento' in selected_features:
        selected_features.remove('segmento')
    print(f"[+] Features selected [{len(selected_features)}]:\n{selected_features}")

    data = data[selected_features]
    train_data, train_rpta, test_data, test_rpta = split_train_test(data, rpta)

    if GRID_SEARCH:
        print("[!] CV / GridSearchCV")
        clf = xgb.XGBClassifier()
        grid = GridSearchCV(clf, GRID_PARAMS, cv=3, scoring="neg_log_loss", n_jobs=-1)
        bal_data, bal_rpta = balance_data(data, rpta, max_ones=1000)
        grid.fit(bal_data, bal_rpta)
        print("[!] Grid Search:")
        print(f"\tBest params: {grid.best_params_}")
        print(f"\tBest score: {grid.best_score_}")
        PARAM.update(grid.best_params_)

    bal_data, bal_rpta = balance_data(train_data, train_rpta, balance_factor=_BALANCE_FACTOR)
    d_train = xgb.DMatrix(bal_data, label=bal_rpta)
    model = xgb.train(PARAM, d_train, NUM_ROUND)
    # model = xgb.XGBRegressor(
    #     n_estimators=100,
    #     learning_rate=.1,
    #     max_depth=6,
    #     random_state=42,
    #     n_jobs=-1,
    #     early_stopping_rounds=10
    # )
    # model.fit(
    #     bal_data, bal_rpta,
    #     eval_metric="rmse",
    #     eval_set=[(test_data, test_rpta)],
    #     verbose=True
    # )

    # model_loaded = False
    # if _LOAD_MODEL:
    #     try:
    #         model = keras.models.load_model('data/model.h5')
    #         model_loaded = True
    #     except FileNotFoundError:
    #         print("[!] Could not load model from data/model.h5")

    # if not model_loaded:
    #     model = base_model(input_dim=len(data.columns))
    #     model.summary()
    #     for epoch in range(_EPOCHS):
    #         fit = model.fit(
    #             bal_data, bal_rpta,
    #             epochs=1, batch_size=_BATCH_SIZE,
    #             validation_split=0.2,
    #             # validation_data=(test_data, test_rpta),
    #             verbose=2,
    #         )
    #         print(f"Epoch {epoch}/{_EPOCHS}")
    #         if 'val_acc' in fit.history and fit.history['val_acc'][-1] >= VAL_ACC_TH:
    #             break

    #     model.save('data/model.h5')

    # test_loss, test_acc = model.evaluate(test_data, test_rpta, verbose=2)
    # print('\nTest loss:', test_loss, '\nTest accuracy:', test_acc)

    # High voltage
    for max_ones in (50, 250, 900, 1400, 2000):
        print(f"-- [ ones: {max_ones} ]")
        bal_data, bal_rpta = balance_data(
            test_data, test_rpta, max_ones=max_ones, balance_factor=1.0)
        # prediction = model.predict_classes(bal_data)
        d_test = xgb.DMatrix(bal_data)
        prediction = model.predict(d_test)
        prediction = pd.DataFrame(
            [round(value) for value in prediction],
            # prediction[:, 0],
            columns=['probabilidad'],
            index=bal_data.index
        )

        ones = prediction[prediction.probabilidad == 1]
        print('Predicted', len(ones), 'ones. Expected', bal_rpta[bal_rpta == 1].count())
        wrong = np.where(prediction.probabilidad != bal_rpta)[0]
        print(f"Error: {len(wrong) / len(bal_rpta)} ({len(wrong)} of {len(bal_rpta)})\n")

    # ================== [PREDICT] ================== #
    if PREDICT:
        print("[!] Predicting real data")
        ids, data, _ = load_data(training=False, cache=_CACHE)
        data = data[selected_features]
        d_test = xgb.DMatrix(data)
        prediction = model.predict(d_test)
        submission = pd.DataFrame(prediction, columns=['probabilidad'], index=ids)
        submission.to_csv('data/submissions/jose.csv')

    con.close()
