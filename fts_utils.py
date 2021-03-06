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
            sum(CASE WHEN hora_trxn BETWEEN 9 AND 18 THEN 1 ELSE 0 END) AS total_office_trxn,
            sum(CASE WHEN hora_trxn BETWEEN 9 AND 18 THEN 0 ELSE 1 END) AS total_night_trxn,
            sum(CASE WHEN dia_trxn BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS total_weekday_trxn,
            sum(CASE WHEN dia_trxn BETWEEN 1 AND 5 THEN 0 ELSE 1 END) AS total_weekend_trxn,
            sum(CASE WHEN month_trxn IN (1, 6, 7, 12) THEN 1 ELSE 0 END) AS high_season_trxn,
            sum(CASE WHEN month_trxn IN (1, 6, 7, 12) THEN 0 ELSE 1 END) AS low_season_trxn
        FROM (
            SELECT id,
                EXTRACT(hour FROM fecha_trxn) AS hora_trxn,
                EXTRACT(dow FROM fecha_trxn) AS dia_trxn,
                EXTRACT(month FROM fecha_trxn) AS month_trxn
            FROM {0}
        ) AS trxn_ts
        GROUP BY id
        ORDER BY id
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_month_entropy(con, data_table_name):
    query = """
        SELECT id,
            -(
                prob_trxn_jan * (CASE WHEN prob_trxn_jan = 0 THEN 0 ELSE log(prob_trxn_jan) END)
                + prob_trxn_feb * (CASE WHEN prob_trxn_feb = 0 THEN 0 ELSE log(prob_trxn_feb) END)
                + prob_trxn_mar * (CASE WHEN prob_trxn_mar = 0 THEN 0 ELSE log(prob_trxn_mar) END)
                + prob_trxn_apr * (CASE WHEN prob_trxn_apr = 0 THEN 0 ELSE log(prob_trxn_apr) END)
                + prob_trxn_may * (CASE WHEN prob_trxn_may = 0 THEN 0 ELSE log(prob_trxn_may) END)
                + prob_trxn_jun * (CASE WHEN prob_trxn_jun = 0 THEN 0 ELSE log(prob_trxn_jun) END)
                + prob_trxn_jul * (CASE WHEN prob_trxn_jul = 0 THEN 0 ELSE log(prob_trxn_jul) END)
                + prob_trxn_aug * (CASE WHEN prob_trxn_aug = 0 THEN 0 ELSE log(prob_trxn_aug) END)
                + prob_trxn_sep * (CASE WHEN prob_trxn_sep = 0 THEN 0 ELSE log(prob_trxn_sep) END)
                + prob_trxn_oct * (CASE WHEN prob_trxn_oct = 0 THEN 0 ELSE log(prob_trxn_oct) END)
                + prob_trxn_nov * (CASE WHEN prob_trxn_nov = 0 THEN 0 ELSE log(prob_trxn_nov) END)
                + prob_trxn_dec * (CASE WHEN prob_trxn_dec = 0 THEN 0 ELSE log(prob_trxn_dec) END)
            ) AS month_entropy
        FROM (
            SELECT id,
                sum(CASE WHEN month_trxn = 1 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_jan,
                sum(CASE WHEN month_trxn = 2 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_feb,
                sum(CASE WHEN month_trxn = 3 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_mar,
                sum(CASE WHEN month_trxn = 4 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_apr,
                sum(CASE WHEN month_trxn = 5 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_may,
                sum(CASE WHEN month_trxn = 6 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_jun,
                sum(CASE WHEN month_trxn = 7 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_jul,
                sum(CASE WHEN month_trxn = 8 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_aug,
                sum(CASE WHEN month_trxn = 9 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_sep,
                sum(CASE WHEN month_trxn = 10 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_oct,
                sum(CASE WHEN month_trxn = 11 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_nov,
                sum(CASE WHEN month_trxn = 12 THEN 1 ELSE 0 END)::numeric / count(*) AS prob_trxn_dec
            FROM (
                SELECT id, EXTRACT(month FROM fecha_trxn) AS month_trxn FROM {0}
            ) AS trxn_ts
            GROUP BY id
            ORDER BY id
        ) AS prob_mes
        ORDER BY id
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_horadia_entropy(con, data_table_name):
    query = """
        SELECT id,
            -(
                prob_office_weekday_trxn * (CASE WHEN prob_office_weekday_trxn = 0 THEN 0 ELSE log(prob_office_weekday_trxn) END)
                + prob_night_weekday_trxn * (CASE WHEN prob_night_weekday_trxn = 0 THEN 0 ELSE log(prob_night_weekday_trxn) END)
                + prob_office_weekend_trxn * (CASE WHEN prob_office_weekend_trxn = 0 THEN 0 ELSE log(prob_office_weekend_trxn) END)
                + prob_night_weekend_trxn * (CASE WHEN prob_night_weekend_trxn = 0 THEN 0 ELSE log(prob_night_weekend_trxn) END)
            ) AS trxn_horadia_entropy
        FROM (
            SELECT id,
                sum(CASE
                    WHEN (hora_trxn BETWEEN 9 AND 18)
                    AND (dia_trxn BETWEEN 1 AND 5)
                    THEN 1 ELSE 0 END
                )::numeric / count(*) AS prob_office_weekday_trxn,
                sum(CASE
                    WHEN (hora_trxn NOT BETWEEN 9 AND 18)
                    AND (dia_trxn BETWEEN 1 AND 5)
                    THEN 1 ELSE 0 END
                )::numeric / count(*) AS prob_night_weekday_trxn,
                sum(CASE
                    WHEN (hora_trxn BETWEEN 9 AND 18)
                    AND (dia_trxn NOT BETWEEN 1 AND 5)
                    THEN 1 ELSE 0 END
                )::numeric / count(*) AS prob_office_weekend_trxn,
                sum(CASE
                    WHEN (hora_trxn NOT BETWEEN 9 AND 18)
                    AND (dia_trxn NOT BETWEEN 1 AND 5)
                    THEN 1 ELSE 0 END
                )::numeric / count(*) AS prob_night_weekend_trxn
            FROM (
                SELECT id,
                    EXTRACT(hour FROM fecha_trxn) AS hora_trxn,
                    EXTRACT(dow FROM fecha_trxn) AS dia_trxn
                FROM {0}
            ) AS trxn_ts
            GROUP BY id
            ORDER BY id
        ) AS prob_horadia
        ORDER BY id
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_culpabanco_entropy(con, data_table_name):
    query = """
        SELECT id,
            -(
                prob_culpa_banco * (CASE WHEN prob_culpa_banco = 0 THEN 0 ELSE log(prob_culpa_banco) END)
                + prob_no_culpa_banco * (CASE WHEN prob_no_culpa_banco = 0 THEN 0 ELSE log(prob_no_culpa_banco) END)
            ) AS culpabanco_entropy
        FROM (
            SELECT id,
                sum(CASE WHEN culpa_banco = 'SI' THEN 1 ELSE 0 END)::numeric / count(*) AS prob_culpa_banco,
                sum(CASE WHEN culpa_banco = 'SI' THEN 0 ELSE 1 END)::numeric / count(*) AS prob_no_culpa_banco
            FROM {0} as trxns
            INNER JOIN {1} AS maestro
                ON trxns.canal = maestro.canal
                    AND trxns.disposit = maestro.disposit
                    AND trxns.cdgtrn = maestro.cdgtrn
                    AND trxns.cdgrpta = maestro.cdgrpta
            GROUP BY id
            ORDER BY id
        ) AS prob
        ORDER BY id
    """.format(data_table_name, maestro_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')


def fts_interval(con, data_table_name):
    query = """
        SELECT id,
            EXTRACT(epoch from avg(trxn_interval)) as avg_trxn_interval
        FROM (
            SELECT id,
                (LEAD(fecha_trxn, 1) OVER (PARTITION BY id ORDER BY id)) - fecha_trxn AS trxn_interval
            FROM {0}
            GROUP BY id, fecha_trxn
            ORDER BY id, fecha_trxn
        ) AS intervals
        GROUP BY id
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


def fts_umbrales(con, data_table_name):
    query = """
        SELECT id,
            -(
                prob_trxn_lun * (CASE WHEN prob_trxn_lun = 0 THEN 0 ELSE log(prob_trxn_lun) END)
                + prob_trxn_mar * (CASE WHEN prob_trxn_mar = 0 THEN 0 ELSE log(prob_trxn_mar) END)
                + prob_trxn_mie * (CASE WHEN prob_trxn_mie = 0 THEN 0 ELSE log(prob_trxn_mie) END)
                + prob_trxn_jue * (CASE WHEN prob_trxn_jue = 0 THEN 0 ELSE log(prob_trxn_jue) END)
                + prob_trxn_vie * (CASE WHEN prob_trxn_vie = 0 THEN 0 ELSE log(prob_trxn_vie) END)
                + prob_trxn_sab * (CASE WHEN prob_trxn_sab = 0 THEN 0 ELSE log(prob_trxn_sab) END)
                + prob_trxn_dom * (CASE WHEN prob_trxn_dom = 0 THEN 0 ELSE log(prob_trxn_dom) END)
            ) AS trxn_week_entropy,
            umbral_trxn_lun,
            umbral_trxn_mar,
            umbral_trxn_mie,
            umbral_trxn_jue,
            umbral_trxn_vie,
            umbral_trxn_sab,
            umbral_trxn_dom
        FROM (
            SELECT id,
                sum(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 0 THEN total_trxn_sesion ELSE 0 END)::numeric / sum(total_trxn_sesion) AS prob_trxn_lun,
                sum(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 1 THEN total_trxn_sesion ELSE 0 END)::numeric / sum(total_trxn_sesion) AS prob_trxn_mar,
                sum(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 2 THEN total_trxn_sesion ELSE 0 END)::numeric / sum(total_trxn_sesion) AS prob_trxn_mie,
                sum(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 3 THEN total_trxn_sesion ELSE 0 END)::numeric / sum(total_trxn_sesion) AS prob_trxn_jue,
                sum(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 4 THEN total_trxn_sesion ELSE 0 END)::numeric / sum(total_trxn_sesion) AS prob_trxn_vie,
                sum(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 5 THEN total_trxn_sesion ELSE 0 END)::numeric / sum(total_trxn_sesion) AS prob_trxn_sab,
                sum(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 6 THEN total_trxn_sesion ELSE 0 END)::numeric / sum(total_trxn_sesion) AS prob_trxn_dom,

                avg(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 0 THEN total_trxn_sesion END)::numeric
                    + 3*stddev(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 0 THEN total_trxn_sesion END)::numeric AS umbral_trxn_lun,
                avg(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 1 THEN total_trxn_sesion END)::numeric
                    + 3*stddev(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 1 THEN total_trxn_sesion END)::numeric AS umbral_trxn_mar,
                avg(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 2 THEN total_trxn_sesion END)::numeric
                    + 3*stddev(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 2 THEN total_trxn_sesion END)::numeric AS umbral_trxn_mie,
                avg(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 3 THEN total_trxn_sesion END)::numeric
                    + 3*stddev(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 3 THEN total_trxn_sesion END)::numeric AS umbral_trxn_jue,
                avg(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 4 THEN total_trxn_sesion END)::numeric
                    + 3*stddev(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 4 THEN total_trxn_sesion END)::numeric AS umbral_trxn_vie,
                avg(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 5 THEN total_trxn_sesion END)::numeric
                    + 3*stddev(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 5 THEN total_trxn_sesion END)::numeric AS umbral_trxn_sab,
                avg(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 6 THEN total_trxn_sesion END)::numeric
                    + 3*stddev(CASE WHEN EXTRACT(dow FROM avg_fecha_sesion) = 6 THEN total_trxn_sesion END)::numeric AS umbral_trxn_dom
            FROM (
                SELECT id, sesion,
                    to_timestamp(avg(extract(epoch from fecha_trxn))) AT TIME ZONE 'UTC' AS avg_fecha_sesion,
                    count(fecha_trxn) AS total_trxn_sesion
                FROM {0}
                GROUP BY id, sesion
                ORDER BY id, sesion
            ) AS tb
            GROUP BY id
            ORDER BY id
        ) AS prob_mes
        ORDER BY id
    """.format(data_table_name)
    return pd.read_sql_query(query, con=con).set_index('id')
