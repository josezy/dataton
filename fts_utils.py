import pandas as pd

from stringcase import snakecase


maestro_table_name = 'maestro_cdgtrn_cdgrpta'


def fts_totales(con, data_table_name):
    query = """
        SELECT id,
            count(id) AS total_trxn,
            count(distinct sesion) AS total_sesiones
        FROM {0}
        GROUP BY id
        ORDER BY id;
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_var_rpta(con, rpta_table_name, training):
    var_rpta_value = ', var_rpta' if training else ''
    query = """
        SELECT id, f_analisis, segmento{1}
        FROM {0}
        ORDER BY id;
    """.format(rpta_table_name, var_rpta_value)
    var_rpta = pd.read_sql_query(query, con=con).set_index('id')

    # remove month part of f_analisis
    var_rpta['year_analisis'] = var_rpta.apply(
        lambda row: int(str(int(row.f_analisis))[:4]),
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
    }).reset_index().set_index('id')
    canales = sesiones.canal.value_counts().unstack()
    canal = pd.DataFrame({
        'times_used_canal': canales.max(axis=1),
    }).reset_index().set_index('id')
    return disposit, canal


def fts_sesion_stats(con, data_table_name):
    query = """
        SELECT id,
            sum(CASE WHEN atypical_sesion = 1 THEN 1 ELSE 0 END) AS total_atypical_sesion,
            EXTRACT(epoch FROM avg(CASE WHEN atypical_sesion = 0 THEN sesion_length END)) AS avg_normal_sesion
        FROM (
            SELECT id,
                sesion,
                (max(fecha_trxn) - min(fecha_trxn)) as sesion_length,
                (CASE
                    WHEN max(fecha_trxn) - min(fecha_trxn) > '1 hour'::interval THEN 1 ELSE 0
                END) AS atypical_sesion
            FROM {0}
            GROUP BY id, sesion
            ORDER BY id, sesion
        ) AS sesion_stats
        GROUP BY id
        ORDER BY id
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_modas(con, data_table_name):
    query = """
        SELECT id,
            mode() WITHIN GROUP (ORDER BY cdgtrn) AS moda_cdgtrn,
            mode() WITHIN GROUP (ORDER BY cdgrpta) AS moda_cdgrpta
        FROM {0}
        GROUP BY id
        ORDER BY id
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_ratio(con, data_table_name):
    query = """
        SELECT trxns.id,
            (sum(CASE WHEN mcc.clasif_trxn = 'Financiera' THEN 1 ELSE 0 END)::numeric / count(*)) AS ratio_financiera,
            (sum(CASE WHEN mcc.clasif_cod_rpta = 'Exitosa' THEN 1 ELSE 0 END)::numeric / count(*)) AS ratio_exitosa,
            (sum(CASE WHEN mcc.clasif_cod_rpta = 'No Exitosa' THEN 1 ELSE 0 END)::numeric / count(*)) AS ratio_no_exitosa
        FROM {0} AS trxns
        INNER JOIN {1} AS mcc
            ON trxns.canal = mcc.canal
                AND trxns.disposit = mcc.disposit
                AND trxns.cdgtrn = mcc.cdgtrn
                AND trxns.cdgrpta = mcc.cdgrpta
        GROUP BY trxns.id
        ORDER BY trxns.id
    """.format(data_table_name, maestro_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_fecha_trxn(con, data_table_name):
    query = """
        SELECT id,
            sum(CASE WHEN hora_trxn between 9 and 18 THEN 1 ELSE 0 END) AS total_office_trxn,
            sum(CASE WHEN hora_trxn between 9 and 18 THEN 0 ELSE 1 END) AS total_night_trxn,
            sum(CASE WHEN dia_trxn between 1 and 5 THEN 1 ELSE 0 END) AS total_weekday_trxn,
            sum(CASE WHEN dia_trxn between 1 and 5 THEN 0 ELSE 1 END) AS total_weekend_trxn
        FROM (
            SELECT id,
                EXTRACT(hour FROM fecha_trxn) AS hora_trxn,
                EXTRACT(dow FROM fecha_trxn) AS dia_trxn
            FROM {0}
        ) AS trxn_ts
        GROUP BY id
        ORDER BY id
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_ratios_maestro(con, data_table_name):
    descripcion_grupos = pd.read_sql_query(
        "SELECT DISTINCT descripcion_grupo FROM {0}".format(maestro_table_name), con=con)
    d_grupo_values = descripcion_grupos.values
    d_grupo_ratios = ','.join("""
        (sum(CASE WHEN mcc.descripcion_grupo = '{0}' THEN 1 ELSE 0 END)/count(*)) AS ratio_dg_{1}
    """.format(d_grupo, snakecase(
        d_grupo.lower().replace('/', '').replace('#', '').replace('  ', ' ')
    )) for d_grupo in d_grupo_values[:, 0])

    producto_asociados = pd.read_sql_query(
        "SELECT DISTINCT producto_asociado FROM {0}".format(maestro_table_name), con=con)
    prod_asociado_values = producto_asociados.values
    prod_asociado_ratios = ','.join("""
        (sum(CASE WHEN mcc.producto_asociado = '{0}' THEN 1 ELSE 0 END)/count(*)) AS ratio_pa_{1}
    """.format(prod_asociado, snakecase(
        prod_asociado.lower().replace('/', '').replace('#', '').replace('  ', ' ')
    )) for prod_asociado in prod_asociado_values[:, 0])

    query = """
        SELECT id, {2}, {3}
        FROM {0} AS trxns
        INNER JOIN {1} AS mcc
            ON trxns.canal = mcc.canal
                AND trxns.disposit = mcc.disposit
                AND trxns.cdgtrn = mcc.cdgtrn
                AND trxns.cdgrpta = mcc.cdgrpta
        GROUP BY trxns.id
        ORDER BY trxns.id
    """.format(data_table_name, maestro_table_name, d_grupo_ratios, prod_asociado_ratios)
    return pd.read_sql_query(query, con=con).set_index('id')
