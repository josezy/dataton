import psycopg2

import numpy as np
import pandas as pd

from tensorflow import keras


LITE = True
PREDICT = False
_BALANCE_FACTOR = 2.0  # relation between number of 0 & 1

con = psycopg2.connect(
    database="dataton",
    user="dataton",
    password="dataton",
    host="localhost",
    port="5433"
)


def load_data(training=True, cache=True):
    print(f"[!] Loading data: training={training}, lite={LITE}, cache={cache}")
    train_str = '_train' if training else '_predict'
    lite_str = '_lite' if LITE else ''
    feats_file_name = f"features{lite_str}{train_str}.csv"

    if cache:
        try:
            fts = pd.read_csv(f'data/{feats_file_name}')
            return fts.id, fts.iloc[:, 1:-1], fts.var_rpta
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

    print("total_trxn, total_sesiones")
    query = """
        SELECT id,
            count(id) AS total_trxn,
            count(distinct sesion) AS total_sesiones
        FROM {0}
        GROUP BY id
        ORDER BY id;
    """.format(data_table_name)
    totales = pd.read_sql_query(query, con=con)

    print("month_analisis, segmento")
    var_rpta_value = ', var_rpta' if training else ''
    query = """
        SELECT id, f_analisis, segmento{1}
        FROM {0}
        ORDER BY id;
    """.format(rpta_table_name, var_rpta_value)
    var_rpta = pd.read_sql_query(query, con=con)

    # remove year part of f_analisis
    var_rpta['month_analisis'] = var_rpta.apply(
        lambda row: int(str(int(row.f_analisis))[4:]),
        axis=1
    )

    print("times_used_disp, times_used_canal")
    query = """
        SELECT DISTINCT ON(id, sesion) *
        FROM {0}
    """.format(data_table_name)
    sesiones = pd.read_sql_query(query, con=con).groupby(['id'])

    disps = sesiones.disposit.value_counts().unstack()
    disposit = pd.DataFrame({
        'times_used_disp': disps.max(axis=1),
    }).reset_index()
    canales = sesiones.canal.value_counts().unstack()
    canal = pd.DataFrame({
        'times_used_canal': canales.max(axis=1),
    }).reset_index()

    # !! avg_sesion_length, total_sesiones_atipicas
    # query = """
    #     SELECT id,
    #         sesion,
    #         (max(fecha_trxn) - min(fecha_trxn)) as sesion_length
    #     FROM datos_transaccionales_train_lite
    #     GROUP BY id, sesion
    #     ORDER BY id, sesion
    # """

    features = pd.concat([
        totales.total_trxn,
        totales.total_sesiones,
        var_rpta.month_analisis,
        var_rpta.segmento,
        disposit.times_used_disp,
        canal.times_used_canal,
    ], axis=1)
    full_features = pd.concat([
        totales.id,
        features,
        var_rpta.get('var_rpta')
    ], axis=1)
    full_features.to_csv(f'data/{feats_file_name}', index=False)
    return totales.id, features, var_rpta.get('var_rpta')


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


def train_model(model, data, rpta, epochs=100):
    bal_data, bal_rpta = balance_data(data, rpta)
    model.fit(bal_data, bal_rpta, epochs=epochs, batch_size=128)
    test_loss, test_acc = model.evaluate(data, rpta, verbose=2)
    print('\nTest loss:', test_loss)
    print('Test accuracy:', test_acc)


def build_model(input_dim):
    # Model definition
    model = keras.Sequential([
        keras.layers.Dense(12, input_dim=input_dim, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
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
    _, data, rpta = load_data()
    print("[!] Data loaded")

    train_data, train_rpta, test_data, test_rpta = split_train_test(data, rpta)
    model = build_model(input_dim=len(train_data.columns))
    train_model(model, train_data, train_rpta, epochs=100)

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
        prediction = pd.DataFrame(prediction[:, 0], columns=['probabilidad'])
        submission = pd.concat([ids, prediction], axis=1)
        submission.to_csv('data/submissions/jose.csv', index=False)

    con.close()
