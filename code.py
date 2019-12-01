import psycopg2

# import numpy as np
import pandas as pd
# import featuretools as ft
# import composeml as cp

from tensorflow import keras


PREDICT = False
IS_LITE = True
BALANCE_FACTOR = 2.0  # relation between number of 0 & 1

con = psycopg2.connect(
    database="dataton",
    user="dataton",
    password="dataton",
    host="localhost",
    port="5433"
)


def load_data(training=True):
    lite_str = '_lite' if IS_LITE else ''
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

    query = """
        SELECT id,
            count(id) AS total_trxn,
            count(distinct sesion) AS total_sesiones
        FROM {0}
        GROUP BY id
        ORDER BY id;
    """.format(data_table_name)
    totales = pd.read_sql_query(query, con=con)

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

    query = """
        SELECT DISTINCT ON(id, sesion) *
        FROM {0}
    """.format(data_table_name)
    sesiones = pd.read_sql_query(query, con=con).groupby(['id'])

    disps = sesiones.disposit.value_counts().unstack()
    disposit = pd.DataFrame({
        'most_used_disp': disps.idxmax(axis=1),
        'times_used_disp': disps.max(axis=1),
    }).reset_index()
    canales = sesiones.canal.value_counts().unstack()
    canal = pd.DataFrame({
        'most_used_canal': canales.idxmax(axis=1),
        'times_used_canal': canales.max(axis=1),
    }).reset_index()

    features = pd.concat([
        totales.total_trxn,
        totales.total_sesiones,
        var_rpta.month_analisis,
        var_rpta.segmento,
        disposit.most_used_disp.astype('category').cat.codes,
        disposit.times_used_disp,
        canal.most_used_canal.astype('category').cat.codes,
        canal.times_used_canal,
    ], axis=1)

    return totales.id, features, var_rpta.get('var_rpta')


def train_model(model, epochs=500):
    # balance number of 0 and 1's
    max_rptas = int(rpta.value_counts().min() * BALANCE_FACTOR)
    bal_data = pd.concat([
        data[rpta == 0][:int(max_rptas * BALANCE_FACTOR)],
        data[rpta == 1][:max_rptas]
    ])
    bal_rpta = pd.concat([
        rpta[rpta == 0][:int(max_rptas * BALANCE_FACTOR)],
        rpta[rpta == 1][:max_rptas]
    ])
    model.fit(bal_data, bal_rpta, epochs=epochs, batch_size=128)
    test_loss, test_acc = model.evaluate(data, rpta, verbose=2)
    print('\nTest loss:', test_loss)
    print('Test accuracy:', test_acc)


if __name__ == '__main__':
    _, data, rpta = load_data()

    # Model definition
    model = keras.Sequential([
        keras.layers.Dense(12, input_dim=len(data.columns), activation='relu'),
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
    train_model(model, epochs=100)

    # High voltage
    prediction = model.predict_classes(data)
    prediction = pd.DataFrame(prediction[:, 0], columns=['probabilidad'])
    ones = prediction[prediction > 0.5]
    print('\nTotal prediction 1:', ones.count())
    print('Total rpta 1:', rpta[rpta == 1].count())

    wrong = prediction[prediction != rpta]
    print('Total errors:', wrong.count(), 'of', len(rpta), 'clients')

    # ================== [PREDICT] ================== #
    if PREDICT:
        ids, data, _ = load_data(training=False)
        prediction = model.predict(data)
        prediction = pd.DataFrame(prediction[:, 0], columns=['probabilidad'])
        submission = pd.concat([ids, prediction], axis=1)
        submission.to_csv('data/submissions/jose.csv', index=False)

    con.close()
