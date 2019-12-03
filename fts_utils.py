import pandas as pd


def fts_totales(con, data_table_name):
    query = """
        SELECT id,
            count(id) AS total_trxn,
            count(distinct sesion) AS total_sesiones
        FROM {0}
        GROUP BY id
        ORDER BY id;
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con)


def fts_var_rpta(con, rpta_table_name, training):
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
    return var_rpta


def fts_sesiones(con, data_table_name):
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
    return disposit, canal


def fts_sesion_stats(con, data_table_name):
    # !! avg_sesion_length, total_sesiones_atipicas
    # query = """
    #     SELECT id,
    #         sesion,
    #         (max(fecha_trxn) - min(fecha_trxn)) as sesion_length
    #     FROM {0}
    #     GROUP BY id, sesion
    #     ORDER BY id, sesion
    # """.format(data_table_name)
    pass
