import psycopg2

import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from fts_utils import (
    fts_totales,
    fts_var_rpta,
    # fts_sesiones,
    fts_modas,
    fts_ratio,
    fts_sesion_stats,
    fts_fecha_trxn,
    fts_ratios_descripcion_grupo,
)


PREDICT = True
_LITE = False
_CACHE = False
_EPOCHS = 1500
_BALANCE_FACTOR = 1.0  # relation between number of 0 & 1

con = psycopg2.connect(
    database="dataton",
    user="dataton",
    password="dataton",
    host="localhost",
    port="5433"
)


def load_data(training=True, lite=True, cache=True):
    lite = lite if training else False
    print(f"[!] Loading data: training={training}, lite={lite}, cache={cache}")
    train_str = '_train' if training else '_predict'
    lite_str = '_lite' if lite else ''
    feats_file_name = f"features{train_str}{lite_str}.csv"

    if cache:
        try:
            fts = pd.read_csv(f'data/features/{feats_file_name}')
            print("Using cached features from ", feats_file_name)
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

    print("ratio_descripcion_grupo, ratio_producto_asociado")
    d_grupo_ratios = fts_ratios_descripcion_grupo(con, data_table_name)

    print("total_office_trxn, total_night_trxn")
    fecha_trxns = fts_fecha_trxn(con, data_table_name)

    print("total_atypical_sesion, avg_normal_sesion")
    sesiones = fts_sesion_stats(con, data_table_name)

    print("ratio_financiera, ratio_exitosa")
    ratios = fts_ratio(con, data_table_name)

    print("moda_cdgtrn, moda_cdgrpta")
    modas = fts_modas(con, data_table_name)

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
    features = features.join(modas, on='id')
    features = features.join(ratios, on='id')
    features = features.join(sesiones, on='id')
    features = features.join(fecha_trxns, on='id')
    features = features.join(d_grupo_ratios, on='id')

    features = features.fillna(-1)
    scaler = StandardScaler()
    features[features.columns] = scaler.fit_transform(features[features.columns])

    full_features = pd.concat([
        features,
        var_rpta.get('var_rpta')
    ], axis=1)
    full_features.to_csv(f'data/features/{feats_file_name}')
    return var_rpta.index.to_series(), features, var_rpta.get('var_rpta')


def balance_data(data, rpta, max_rptas=None, balance_factor=_BALANCE_FACTOR):
    # balance number of 0 and 1's
    max_rptas = max_rptas or int(rpta.value_counts().min() * balance_factor)
    bal_data = pd.concat([
        data[rpta == 0][:int(max_rptas * balance_factor)],
        data[rpta == 1][:max_rptas]
    ])
    bal_rpta = pd.concat([
        rpta[rpta == 0][:int(max_rptas * balance_factor)],
        rpta[rpta == 1][:max_rptas]
    ])
    return bal_data, bal_rpta


def split_train_test(data, rpta, tt_threshold=0.7):
    threshold = int(len(rpta) * tt_threshold)
    train_data = data.iloc[:threshold]
    test_data = data.iloc[threshold:]
    train_rpta = rpta.iloc[:threshold]
    test_rpta = rpta.iloc[threshold:]
    return train_data, train_rpta, test_data, test_rpta


def build_model(input_dim):
    # Model definition
    model = keras.Sequential([
        keras.layers.Dense(12, input_dim=input_dim, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    _, data, rpta = load_data(lite=_LITE, cache=_CACHE)
    train_data, train_rpta, test_data, test_rpta = split_train_test(data, rpta)
    print("[!] Data loaded")

    model = build_model(input_dim=len(data.columns))

    bal_data, bal_rpta = balance_data(train_data, train_rpta)
    model.fit(bal_data, bal_rpta, epochs=_EPOCHS)

    test_loss, test_acc = model.evaluate(test_data, test_rpta, verbose=2)
    print('\nTest loss:', test_loss, '\nTest accuracy:', test_acc)

    # High voltage
    bal_data, bal_rpta = balance_data(
        test_data, test_rpta, max_rptas=200, balance_factor=1.0)
    prediction = model.predict_classes(bal_data)
    prediction = pd.DataFrame(
        prediction[:, 0],
        columns=['probabilidad'],
        index=bal_data.index
    )
    ones = prediction[prediction.probabilidad == 1]
    print('\nPredicted', len(ones), 'ones. Expected', bal_rpta[bal_rpta == 1].count())

    wrong = np.where(prediction.probabilidad != bal_rpta)[0]
    print(f"Error: {len(wrong) / len(bal_rpta)} ({len(wrong)} of {len(bal_rpta)})\n")

    # ================== [PREDICT] ================== #
    if PREDICT:
        print("[+] Predicting real data")
        ids, data, _ = load_data(training=False, cache=False)
        prediction = model.predict(data)
        submission = pd.DataFrame(prediction[:, 0], columns=['probabilidad'], index=ids)
        submission.to_csv('data/submissions/jose.csv')

    con.close()
