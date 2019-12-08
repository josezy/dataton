-- SQL instructions create tables and load csv data into them

-- TABLES
-- datos_transaccionales_predict    **
-- datos_transaccionales_train      **
-- datos_transaccionales_train_lite **
-- datos_var_rpta_train             **
-- datos_var_rpta_train_lite        **
-- ids_predict                      **
-- maestro_cdgtrn_cdgrpta           **


DROP TABLE ids_predict;
CREATE TABLE ids_predict (
    id integer,
    f_analisis integer,
    segmento integer
);

DROP TABLE datos_transaccionales_predict;
CREATE TABLE datos_transaccionales_predict (
    id integer,
    fecha_trxn timestamp,
    canal varchar,
    disposit varchar,
    cdgtrn integer,
    cdgrpta integer,
    vlrtran numeric,
    sesion integer
);

DROP TABLE datos_transaccionales_train_lite;
CREATE TABLE datos_transaccionales_train_lite (
    id integer,
    fecha_trxn timestamp,
    canal varchar,
    disposit varchar,
    cdgtrn integer,
    cdgrpta integer,
    vlrtran numeric,
    sesion integer
);
CREATE INDEX datos_transaccionales_train_lite_id_index
ON datos_transaccionales_train_lite (id);

DROP TABLE datos_transaccionales_train;
CREATE TABLE datos_transaccionales_train (
    id integer,
    fecha_trxn timestamp,
    canal varchar,
    disposit varchar,
    cdgtrn integer,
    cdgrpta integer,
    vlrtran numeric,
    sesion integer
);
CREATE INDEX datos_transaccionales_train_id_index
ON datos_transaccionales_train (id);

DROP TABLE datos_var_rpta_train_lite;
CREATE TABLE datos_var_rpta_train_lite (
    id integer,
    f_analisis integer,
    var_rpta integer,
    segmento integer
);
CREATE INDEX datos_var_rpta_train_lite_id_index
ON datos_var_rpta_train_lite (id);

DROP TABLE datos_var_rpta_train;
CREATE TABLE datos_var_rpta_train (
    id integer,
    f_analisis integer,
    var_rpta integer,
    segmento integer
);
CREATE INDEX datos_var_rpta_train_id_index
ON datos_var_rpta_train (id);

DROP TABLE maestro_cdgtrn_cdgrpta;
CREATE TABLE maestro_cdgtrn_cdgrpta (
    canal varchar,
    disposit varchar,
    cdgtrn varchar,
    cdgrpta varchar,
    grupo_descrp_trxn varchar,
    descrip_trxn varchar,
    clasif_trxn varchar,
    descripcion_grupo varchar,
    descrip_cod_rpta varchar,
    clasif_cod_rpta varchar,
    grupot_modificado varchar,
    culpa_banco varchar,
    producto_asociado varchar
);


-- Import data from files
COPY ids_predict
FROM '/Users/joseb/dataton/data/DT19_IDs_predict.csv' DELIMITER ',' CSV HEADER;

COPY datos_transaccionales_predict
FROM '/Users/joseb/dataton/data/DT19_Datos_transaccionales_predict.csv' DELIMITER ',' CSV HEADER;

COPY datos_transaccionales_train_lite
FROM '/Users/joseb/dataton/data/DT19_Datos_transaccionales_train_lite.csv' DELIMITER ',' CSV HEADER;

COPY datos_transaccionales_train
FROM '/Users/joseb/dataton/data/DT19_Datos_transaccionales_train.csv' DELIMITER ',' CSV HEADER;

COPY datos_var_rpta_train_lite
FROM '/Users/joseb/dataton/data/DT19_Datos_Var_Rpta_train_lite.csv' DELIMITER ',' CSV HEADER;

COPY datos_var_rpta_train
FROM '/Users/joseb/dataton/data/DT19_Datos_Var_Rpta_train.csv' DELIMITER ',' CSV HEADER;

COPY maestro_cdgtrn_cdgrpta
FROM '/Users/joseb/dataton/data/DT19_maestro_cdgtrn_cdgrpta.csv' DELIMITER ',' CSV HEADER;


-- ARRANGE DATA
-- replace 'None' strings with -1 and change type to integer
UPDATE maestro_cdgtrn_cdgrpta SET cdgtrn = '-1' WHERE cdgtrn = 'None';
UPDATE maestro_cdgtrn_cdgrpta SET cdgrpta = '-1' WHERE cdgrpta = 'None';
ALTER TABLE maestro_cdgtrn_cdgrpta
ALTER COLUMN cdgtrn TYPE integer USING cdgtrn::numeric::integer;
ALTER TABLE maestro_cdgtrn_cdgrpta
ALTER COLUMN cdgrpta TYPE integer USING cdgrpta::numeric::integer;
