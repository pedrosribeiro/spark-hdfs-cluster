from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_date, datediff, col, avg as spark_avg, sum as spark_sum, 
    count as spark_count, stddev as spark_std, first, when
)
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ============ Spark session ============
spark = SparkSession.builder \
    .appName("TPC-H Predict IsLate - Enhanced with Supplier Features") \
    .getOrCreate()

# ============ Paths ============
input_path = "hdfs://namenode:9000/data/input/"
output_path = "hdfs://namenode:9000/data/output/feature_engineered_products/"

# ============ Load TPC-H tables ============
df_lineitem = spark.read.csv(f"{input_path}/lineitem.tbl", sep="|", inferSchema=True, header=False)
df_orders   = spark.read.csv(f"{input_path}/orders.tbl",   sep="|", inferSchema=True, header=False)
df_supplier = spark.read.csv(f"{input_path}/supplier.tbl", sep="|", inferSchema=True, header=False)
df_part     = spark.read.csv(f"{input_path}/part.tbl",     sep="|", inferSchema=True, header=False)
df_customer = spark.read.csv(f"{input_path}/customer.tbl", sep="|", inferSchema=True, header=False)

# ============ Rename columns ============
df_lineitem = df_lineitem.toDF(
    "l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity",
    "l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus",
    "l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct",
    "l_shipmode","l_comment","_empty"
).drop("_empty")

df_orders = df_orders.toDF(
    "o_orderkey","o_custkey","o_orderstatus","o_totalprice","o_orderdate",
    "o_orderpriority","o_clerk","o_shippriority","o_comment","_empty"
).drop("_empty")

df_supplier = df_supplier.toDF(
    "s_suppkey","s_name","s_address","s_nationkey","s_phone",
    "s_acctbal","s_comment","_empty"
).drop("_empty")

df_part = df_part.toDF(
    "p_partkey","p_name","p_mfgr","p_brand","p_type",
    "p_size","p_container","p_retailprice","p_comment","_empty"
).drop("_empty")

df_customer = df_customer.toDF(
    "c_custkey","c_name","c_address","c_nationkey","c_phone",
    "c_acctbal","c_mktsegment","c_comment","_empty"
).drop("_empty")

# ============ Join lineitem + orders + supplier + part + customer ============
df = df_lineitem.join(df_orders, df_lineitem.l_orderkey == df_orders.o_orderkey, how="inner") \
                .join(df_supplier, df_lineitem.l_suppkey == df_supplier.s_suppkey, how="left") \
                .join(df_part, df_lineitem.l_partkey == df_part.p_partkey, how="left") \
                .join(df_customer, df_orders.o_custkey == df_customer.c_custkey, how="left")

# ============ Convert dates ============
df = df.withColumn("l_receiptdate", to_date(col("l_receiptdate"))) \
       .withColumn("l_commitdate", to_date(col("l_commitdate"))) \
       .withColumn("l_shipdate", to_date(col("l_shipdate"))) \
       .withColumn("o_orderdate", to_date(col("o_orderdate")))

# ============ Label creation (allowed: uses l_receiptdate only to compute label) ============
# label: is_late = receiptdate > commitdate
df = df.withColumn("is_late", (col("l_receiptdate") > col("l_commitdate")).cast("integer"))

# ============ Feature engineering (ONLY features known before receipt) ============
# 1. Planned lead time in days: commitdate - orderdate (planning horizon)
df = df.withColumn("planned_lead_days", datediff(col("l_commitdate"), col("o_orderdate")))

# 2. NEW: Lead time do pedido (L_SHIPDATE - O_ORDERDATE) - tempo de envio planejado
df = df.withColumn("order_to_ship_days", datediff(col("l_shipdate"), col("o_orderdate")))

# 3. Tempo de processamento por item: commitdate - orderdate
df = df.withColumn("processing_time_days", datediff(col("l_commitdate"), col("o_orderdate")))

# Drop rows with nulls in critical fields
df = df.dropna(subset=["o_orderdate", "l_commitdate", "l_receiptdate", "l_shipdate", 
                        "planned_lead_days", "l_quantity", "l_discount", "l_suppkey"])

# ============ Supplier historical features (NO DATA LEAKAGE) ============
# CRÍTICO: Para evitar data leakage, usamos Window Functions para calcular
# métricas históricas do fornecedor usando APENAS pedidos ANTERIORES ao pedido atual
print("Calculando features históricas de fornecedor...")

# Window spec: particiona por fornecedor, ordena por data, olha apenas para pedidos anteriores
# rowsBetween(Window.unboundedPreceding, -1) = desde o início até a linha ANTERIOR
# Isso garante que para cada pedido, usamos APENAS o histórico até aquele momento
windowSpec = Window.partitionBy("l_suppkey").orderBy("o_orderdate").rowsBetween(Window.unboundedPreceding, -1)

# Calcular tempo médio de processamento HISTÓRICO por fornecedor (apenas pedidos anteriores)
df = df.withColumn("supplier_avg_processing_time", 
                   spark_avg("processing_time_days").over(windowSpec))

# Calcular frequência de atraso HISTÓRICA por fornecedor (apenas pedidos anteriores)
df = df.withColumn("supplier_late_frequency", 
                   spark_avg("is_late").over(windowSpec))

# Para fornecedores sem histórico (primeiro pedido), preencher com valores neutros
# Usar mediana global ou valores padrão conservadores
df = df.fillna({
    "supplier_avg_processing_time": 30.0,  # valor padrão conservador (30 dias)
    "supplier_late_frequency": 0.5  # 50% (neutro, sem viés)
})

print("Features históricas de fornecedor criadas")

# ============ Aggregate per ORDER (better ML approach) ============
print("Agregando dados por pedido (o_orderkey)...")
agg_df = df.groupBy("o_orderkey").agg(
    # Agregações numéricas
    spark_sum("l_quantity").alias("total_quantity"),
    spark_sum("l_extendedprice").alias("total_extendedprice"),
    spark_avg("l_discount").alias("avg_discount"),
    spark_count("*").alias("num_items"),
    spark_avg("planned_lead_days").alias("avg_planned_lead"),
    spark_std("l_quantity").alias("std_quantity"),
    spark_avg("l_tax").alias("avg_tax"),
    spark_avg("order_to_ship_days").alias("avg_order_to_ship_days"),
    spark_avg("processing_time_days").alias("avg_processing_time"),
    
    # Features de fornecedor (média por pedido)
    spark_avg("supplier_avg_processing_time").alias("avg_supplier_processing"),
    spark_avg("supplier_late_frequency").alias("avg_supplier_late_freq"),
    
    # Valores representativos categóricos (primeiro valor)
    first("l_shipmode").alias("shipmode"),
    first("o_orderpriority").alias("orderpriority"),
    first("o_totalprice").alias("order_total"),
    first("c_mktsegment").alias("customer_segment"),
    
    # Label: proporção de itens atrasados no pedido (late_ratio)
    spark_avg("is_late").alias("late_ratio")
)

# Criar label binário: pedido considerado atrasado se late_ratio > 0
agg_df = agg_df.withColumn("is_late", (col("late_ratio") > 0).cast("integer"))

# Tratar valores nulos na agregação
agg_df = agg_df.fillna({
    "std_quantity": 0.0,
    "avg_tax": 0.0,
    "avg_discount": 0.0,
    "avg_supplier_processing": 0.0,
    "avg_supplier_late_freq": 0.5
})

# ============ Class balance diagnostics ============
counts = agg_df.groupBy("is_late").count().collect()
print("Class distribution (is_late) após agregação:", counts)
print(f"Total de pedidos: {agg_df.count()}")

# ============ Index categorical features ============
indexer_ship = StringIndexer(inputCol="shipmode", outputCol="shipmode_index", handleInvalid="keep")
indexer_prio = StringIndexer(inputCol="orderpriority", outputCol="priority_index", handleInvalid="keep")
indexer_segment = StringIndexer(inputCol="customer_segment", outputCol="segment_index", handleInvalid="keep")

# ============ Assemble feature vector ============
feature_cols = [
    "total_quantity",
    "total_extendedprice",
    "avg_discount",
    "num_items",
    "avg_planned_lead",
    "std_quantity",
    "avg_tax",
    "avg_order_to_ship_days",
    "avg_processing_time",
    "avg_supplier_processing",
    "avg_supplier_late_freq",
    "order_total",
    "shipmode_index",
    "priority_index",
    "segment_index"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

# ============ Train/test split ============
train, test = agg_df.randomSplit([0.8, 0.2], seed=42, )
train = train.withColumn("weight", col("is_late") * 5 + 1)

print(f"Conjunto de treino: {train.count()} pedidos")
print(f"Conjunto de teste: {test.count()} pedidos")

# ============ Pipelines and models ============
# Logistic Regression pipeline
lr = LogisticRegression(featuresCol="features", labelCol="is_late", maxIter=50, regParam=0.01, weightCol="weight")
pipeline_lr = Pipeline(stages=[indexer_ship, indexer_prio, indexer_segment, assembler, lr])

# RandomForest pipeline (increase trees and depth for better performance)
rf = RandomForestClassifier(featuresCol="features", labelCol="is_late", numTrees=20, maxDepth=10, seed=42)
pipeline_rf = Pipeline(stages=[indexer_ship, indexer_prio, indexer_segment, assembler, rf])

# ============ Train ============
print("Training Logistic Regression...")
lr_model = pipeline_lr.fit(train)
print("Training RandomForest...")
rf_model = pipeline_rf.fit(train)

# ============ Predict ============
lr_predictions = lr_model.transform(test)
rf_predictions = rf_model.transform(test)

# ============ Evaluation ============
bce = BinaryClassificationEvaluator(labelCol="is_late", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
mce_acc = MulticlassClassificationEvaluator(labelCol="is_late", predictionCol="prediction", metricName="accuracy")
mce_f1  = MulticlassClassificationEvaluator(labelCol="is_late", predictionCol="prediction", metricName="f1")

def eval_preds(preds, name):
    auc = bce.evaluate(preds)
    acc = mce_acc.evaluate(preds)
    f1  = mce_f1.evaluate(preds)
    return {"model": name, "auc": auc, "accuracy": acc, "f1": f1}

results_lr = eval_preds(lr_predictions, "Logistic Regression")
results_rf = eval_preds(rf_predictions, "RandomForest")

print("\n" + "="*90)
print("RESULTADOS FINAIS - Predição de Atrasos de Pedidos (TPC-H Enhanced)")
print("="*90)
print(f"\nFeatures utilizadas ({len(feature_cols)}):")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {feat}")

print("\n" + "-"*90)
print("Logistic Regression:")
print(f"  AUC-ROC:  {results_lr['auc']:.4f}")
print(f"  Accuracy: {results_lr['accuracy']:.4f}")
print(f"  F1-Score: {results_lr['f1']:.4f}")

print("\n" + "-"*90)
print("Random Forest:")
print(f"  AUC-ROC:  {results_rf['auc']:.4f}")
print(f"  Accuracy: {results_rf['accuracy']:.4f}")
print(f"  F1-Score: {results_rf['f1']:.4f}")

print("\n" + "="*90)
print("Melhorias implementadas:")
print("  ✓ Lead time do pedido (L_SHIPDATE - O_ORDERDATE)")
print("  ✓ Tempo médio de processamento por fornecedor")
print("  ✓ Frequência de atraso por fornecedor")
print("  ✓ Agregação por pedido (order-level)")
print("  ✓ Features de cliente (segmento de mercado)")
print("  ✓ Métricas estatísticas (std, avg) por pedido")
print("="*90 + "\n")

agg_df.repartition(1).write.mode("overwrite").option("header", "true").csv(output_path + "agg_df/")

# ============ End ============
spark.stop()
