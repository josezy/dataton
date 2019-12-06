import time
import psycopg2

import numpy as np
import pandas as pd

from tensorflow import keras
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

from fts_utils import (
    fts_totales,
    fts_var_rpta,
    # fts_sesiones,
    # fts_modas,
    fts_ratio,
    fts_sesion_stats,
    fts_fecha_trxn,
    # fts_ratios_maestro,
    fts_month_entropy,
    fts_horadia_entropy,
)

from ft_selection import rfecv_selector, kbest_selector


VAL_ACC_TH = 0.8
PREDICT = True
_LITE = False
_CACHE = True
_EPOCHS = 1000
_BATCH_SIZE = 128
_LOAD_MODEL = False
_BALANCE_FACTOR = 1.0  # relation between number of 0 & 1


# gs_params = {
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
    bal_data, bal_rpta = balance_data(data, rpta, max_ones=200)
    # return [
    #     'month_entropy',
    #     'total_office_trxn', 'total_night_trxn',
    #     'total_weekday_trxn', 'total_weekend_trxn',
    #     'high_season_trxn', 'low_season_trxn',
    #     'segmento',
    # ]  # Hardcoded features

    # Select by RFECV
    # rfecv_selector.fit(bal_data, bal_rpta)
    # rfecv_features = data.columns[rfecv_selector.support_].to_list()

    # Select by KBest
    kbest_selector.fit_transform(bal_data, bal_rpta)
    f_score_indexes = (-kbest_selector.scores_).argsort()[:16]
    kbest_features = data.columns[f_score_indexes].to_list()

    # Select by correlation matrix
    data = bal_data.join(bal_rpta)
    corrmat = data.corr()
    rpta_corr = abs(corrmat['var_rpta'][:].drop('var_rpta'))
    top = rpta_corr[rpta_corr > rpta_corr.mean()]
    corr_features = top.index.to_list()

    return list(set([
        # *rfecv_selector,
        *kbest_features,
        *corr_features,
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

    print("horadia_entropy")
    horadia_entropy = fts_horadia_entropy(con, data_table_name)

    print("month_entropy")
    month_entropy = fts_month_entropy(con, data_table_name)

    # print("ratio_descripcion_grupo, ratio_producto_asociado")
    # ratios_maestro = fts_ratios_maestro(con, data_table_name)

    print("total_office_trxn, total_night_trxn")
    fecha_trxns = fts_fecha_trxn(con, data_table_name)

    print("total_atypical_sesion, avg_normal_sesion")
    sesiones = fts_sesion_stats(con, data_table_name)

    print("ratio_financiera, ratio_exitosa")
    ratios = fts_ratio(con, data_table_name)

    # print("moda_cdgtrn, moda_cdgrpta")
    # modas = fts_modas(con, data_table_name)

    # print("times_used_disp, times_used_canal")
    # disposit, canal = fts_sesiones(con, data_table_name)

    print("total_trxn, total_sesiones")
    totales = fts_totales(con, data_table_name)

    print("year_analisis, segmento")
    var_rpta = fts_var_rpta(con, rpta_table_name, training)

    features = pd.concat([
        totales.total_trxn,
        totales.total_sesiones,
        # var_rpta.year_analisis,
        var_rpta.segmento,
        # disposit.times_used_disp,
        # canal.times_used_canal,
    ], axis=1)
    # features = features.join(modas, on='id')
    features = features.join(ratios, on='id')
    features = features.join(sesiones, on='id')
    features = features.join(fecha_trxns, on='id')
    # features = features.join(ratios_maestro, on='id')
    features = features.join(month_entropy, on='id')
    features = features.join(horadia_entropy, on='id')

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
    print(f"[+] Features selected [{len(selected_features)}]:\n{selected_features}")

    data = data[selected_features]
    train_data, train_rpta, test_data, test_rpta = split_train_test(data, rpta)
    bal_data, bal_rpta = balance_data(train_data, train_rpta, balance_factor=_BALANCE_FACTOR)

    model_loaded = False
    if _LOAD_MODEL:
        try:
            model = keras.models.load_model('data/model.h5')
            model_loaded = True
        except FileNotFoundError:
            print("[!] Could not load model from data/model.h5")

    if not model_loaded:
        model = base_model(input_dim=len(data.columns))
        model.summary()
        for epoch in range(_EPOCHS):
            fit = model.fit(
                bal_data, bal_rpta,
                epochs=1, batch_size=_BATCH_SIZE,
                validation_split=0.2,
                # validation_data=(test_data, test_rpta),
                verbose=2,
            )
            print(f"Epoch {epoch}/{_EPOCHS}")
            if 'val_acc' in fit.history and fit.history['val_acc'][-1] >= VAL_ACC_TH:
                break

        model.save('data/model.h5')

    test_loss, test_acc = model.evaluate(test_data, test_rpta, verbose=2)
    print('\nTest loss:', test_loss, '\nTest accuracy:', test_acc)

    # High voltage
    for max_ones in (70, 200, 800, 1100, 1600):
        print(f"-- [ ones: {max_ones} ]")
        bal_data, bal_rpta = balance_data(
            test_data, test_rpta, max_ones=max_ones, balance_factor=1.0)
        prediction = model.predict_classes(bal_data)
        prediction = pd.DataFrame(
            prediction[:, 0],
            columns=['probabilidad'],
            index=bal_data.index
        )
        ones = prediction[prediction.probabilidad == 1]
        print('Predicted', len(ones), 'ones. Expected', bal_rpta[bal_rpta == 1].count())

        wrong = np.where(prediction.probabilidad != bal_rpta)[0]
        print(f"Error: {len(wrong) / len(bal_rpta)} ({len(wrong)} of {len(bal_rpta)})\n")

    # def build_model(optimizer='sgd'):
    #     return base_model(input_dim=len(data.columns), optimizer=optimizer)

    # Cross validation
    # classifier = KerasClassifier(build_fn=build_model, nb_epoch=100, batch_size=10)
    # accuracies = cross_val_score(estimator=classifier, X=bal_data, y=bal_rpta, cv=10, n_jobs=-1)
    # print(f"[!] Accuracies: mean={accuracies.mean()}, var={accuracies.var()}")

    # Grid Search
    # classifier = KerasClassifier(build_fn=build_model)
    # grid_search = GridSearchCV(
    #     estimator=classifier, param_grid=gs_params, scoring='accuracy', cv=10)
    # grid_search = grid_search.fit(bal_data, bal_rpta)
    # print("[!] Gris Search:")
    # print(f"\tBest params: {grid_search.best_params_}")
    # print(f"\tBest score: {grid_search.best_score_}")

    # ================== [PREDICT] ================== #
    if PREDICT:
        print("[!] Predicting real data")
        ids, data, _ = load_data(training=False, cache=_CACHE)
        data = data[selected_features]
        prediction = model.predict(data)
        submission = pd.DataFrame(prediction[:, 0], columns=['probabilidad'], index=ids)
        submission.to_csv('data/submissions/jose.csv')

    con.close()
