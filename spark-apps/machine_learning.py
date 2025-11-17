from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_date, datediff, col, sum as spark_sum, 
    avg as spark_avg, when, abs as spark_abs
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ============ Spark session ============
spark = SparkSession.builder \
    .appName("TPC-H Predict IsLate - Simplified") \
    .getOrCreate()

# ============ Paths ============
input_path = "hdfs://namenode:9000/data/input/"
output_path = "hdfs://namenode:9000/data/output/ml_predictions/"

# ============ Load TPC-H tables ============
print("Carregando tabelas TPC-H...")
df_lineitem = spark.read.csv(f"{input_path}/lineitem.tbl", sep="|", inferSchema=True, header=False)
df_orders   = spark.read.csv(f"{input_path}/orders.tbl",   sep="|", inferSchema=True, header=False)
df_supplier = spark.read.csv(f"{input_path}/supplier.tbl", sep="|", inferSchema=True, header=False)
df_customer = spark.read.csv(f"{input_path}/customer.tbl", sep="|", inferSchema=True, header=False)
df_nation   = spark.read.csv(f"{input_path}/nation.tbl",   sep="|", inferSchema=True, header=False)

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

df_customer = df_customer.toDF(
    "c_custkey","c_name","c_address","c_nationkey","c_phone",
    "c_acctbal","c_mktsegment","c_comment","_empty"
).drop("_empty")

df_nation = df_nation.toDF(
    "n_nationkey","n_name","n_regionkey","n_comment","_empty"
).drop("_empty")

print("✓ Tabelas carregadas e renomeadas")

# ============ Join tables ============
print("Fazendo join das tabelas...")
df = df_lineitem \
    .join(df_orders, df_lineitem.l_orderkey == df_orders.o_orderkey, how="inner") \
    .join(df_supplier, df_lineitem.l_suppkey == df_supplier.s_suppkey, how="inner") \
    .join(df_customer, df_orders.o_custkey == df_customer.c_custkey, how="inner")

print("Join concluído")

# ============ Convert dates ============
df = df.withColumn("l_receiptdate", to_date(col("l_receiptdate"))) \
       .withColumn("l_commitdate", to_date(col("l_commitdate"))) \
       .withColumn("o_orderdate", to_date(col("o_orderdate")))

# ============ Label creation ============
# Label: is_late = receiptdate > commitdate
df = df.withColumn("is_late", (col("l_receiptdate") > col("l_commitdate")).cast("integer"))

# ============ Feature 1: Distância Logística (mesmo país = 0, países diferentes = 1) ============
print("Criando feature: distância logística...")
df = df.withColumn("logistic_distance", 
    when(col("s_nationkey") == col("c_nationkey"), 0).otherwise(1)
)

# ============ Feature 2: Expected Lead Time (COMMITDATE - ORDERDATE) ============
print("Criando feature: expected_lead_time...")
df = df.withColumn("expected_lead_time", 
    datediff(col("l_commitdate"), col("o_orderdate"))
)

# ============ Drop rows with nulls ============
df = df.dropna(subset=["o_orderdate", "l_commitdate", "l_receiptdate", 
                        "l_quantity", "expected_lead_time", "s_nationkey", "c_nationkey"])

print(f"Features criadas. Total de linhas: {df.count()}")

# ============ Aggregate per ORDER ============
print("Agregando por pedido (o_orderkey)...")
agg_df = df.groupBy("o_orderkey").agg(
    # Feature 3: Carga total do pedido (soma das quantidades)
    spark_sum("l_quantity").alias("total_volume"),
    
    # Médias das outras features (para consolidar múltiplos itens do mesmo pedido)
    spark_avg("expected_lead_time").alias("avg_expected_lead_time"),
    spark_avg("logistic_distance").alias("avg_logistic_distance"),
    
    # Label: proporção de itens atrasados no pedido
    spark_avg("is_late").alias("late_ratio")
)

# Criar label binário: pedido atrasado se late_ratio > 0.5
agg_df = agg_df.withColumn("is_late", (col("late_ratio") > 0.5).cast("integer"))

# Remover valores nulos
agg_df = agg_df.dropna()

print(f"✓ Agregação concluída. Total de pedidos: {agg_df.count()}")

# ============ Class balance diagnostics ============
counts = agg_df.groupBy("is_late").count().collect()
print(f"\nDistribuição de classes:")
for row in counts:
    label = "Atrasados" if row["is_late"] == 1 else "No prazo"
    print(f"  {label}: {row['count']} pedidos")

# ============ Feature vector assembly ============
feature_cols = [
    "total_volume",              # Feature 1: Carga total do pedido
    "avg_expected_lead_time",    # Feature 2: Tempo prometido
    "avg_logistic_distance"      # Feature 3: Distância logística
]

print(f"\nFeatures utilizadas ({len(feature_cols)}):")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i}. {feat}")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

agg_df.repartition(1).write.mode("overwrite").option("header", "true").csv(
    output_path + "agg_df/"
)

# ============ Train/test split ============
train, test = agg_df.randomSplit([0.8, 0.2], seed=42)
print(f"\nConjunto de treino: {train.count()} pedidos")
print(f"Conjunto de teste: {test.count()} pedidos")

# ============ Models ============
# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="is_late", maxIter=50)
pipeline_lr = Pipeline(stages=[assembler, lr])

# Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="is_late", 
                            numTrees=20, maxDepth=10, seed=42)
pipeline_rf = Pipeline(stages=[assembler, rf])

# ============ Train ============
print("\n" + "="*70)
print("Treinando modelos...")
print("="*70)

print("→ Logistic Regression...")
lr_model = pipeline_lr.fit(train)
print("  ✓ Concluído")

print("→ Random Forest...")
rf_model = pipeline_rf.fit(train)
print("  ✓ Concluído")

# ============ Predict ============
lr_predictions = lr_model.transform(test)
rf_predictions = rf_model.transform(test)

# ============ Evaluation ============
bce = BinaryClassificationEvaluator(labelCol="is_late", rawPredictionCol="rawPrediction", 
                                    metricName="areaUnderROC")
mce_acc = MulticlassClassificationEvaluator(labelCol="is_late", predictionCol="prediction", 
                                            metricName="accuracy")
mce_f1 = MulticlassClassificationEvaluator(labelCol="is_late", predictionCol="prediction", 
                                           metricName="f1")

def eval_model(predictions, model_name):
    auc = bce.evaluate(predictions)
    acc = mce_acc.evaluate(predictions)
    f1 = mce_f1.evaluate(predictions)
    return {"model": model_name, "auc": auc, "accuracy": acc, "f1": f1}

results_lr = eval_model(lr_predictions, "Logistic Regression")
results_rf = eval_model(rf_predictions, "Random Forest")

# ============ Results Summary ============
print("\n" + "="*70)
print("RESULTADOS FINAIS - Predição de Atrasos")
print("="*70)

print("\nFeatures utilizadas:")
print("  1. total_volume - Carga total do pedido")
print("  2. avg_expected_lead_time - Tempo entre pedido e data prometida")
print("  3. avg_logistic_distance - Distância logística (mesmo país = 0)")

print("\n" + "-"*70)
print("Logistic Regression:")
print(f"  AUC-ROC:  {results_lr['auc']:.4f}")
print(f"  Accuracy: {results_lr['accuracy']:.4f}")
print(f"  F1-Score: {results_lr['f1']:.4f}")

print("\n" + "-"*70)
print("Random Forest:")
print(f"  AUC-ROC:  {results_rf['auc']:.4f}")
print(f"  Accuracy: {results_rf['accuracy']:.4f}")
print(f"  F1-Score: {results_rf['f1']:.4f}")

print("\n" + "="*70 + "\n")

# ============ Optional: Save predictions ============
print("Salvando predições no HDFS...")
lr_predictions.select("o_orderkey", "is_late", "prediction", "probability") \
    .write.csv(f"{output_path}/lr_predictions", header=True, mode="overwrite")

rf_predictions.select("o_orderkey", "is_late", "prediction", "probability") \
    .write.csv(f"{output_path}/rf_predictions", header=True, mode="overwrite")

print(f"✓ Predições salvas em {output_path}")

# ============ End ============
spark.stop()
