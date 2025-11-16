from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, datediff, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ============ Spark session ============
spark = SparkSession.builder \
    .appName("TPC-H Predict IsLate (no leakage)") \
    .getOrCreate()

# ============ Paths ============
input_path = "hdfs://namenode:9000/data/input/"
output_path = "hdfs://namenode:9000/data/output/feature_engineered_products/"

# ============ Load TPC-H tables ============
df_lineitem = spark.read.csv(f"{input_path}/lineitem.tbl", sep="|", inferSchema=True, header=False)
df_orders   = spark.read.csv(f"{input_path}/orders.tbl",   sep="|", inferSchema=True, header=False)

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

# ============ Join ============
df = df_lineitem.join(df_orders, df_lineitem.l_orderkey == df_orders.o_orderkey, how="inner")

# ============ Label creation (allowed: uses l_receiptdate only to compute label) ============
df = df.withColumn("l_receiptdate", to_date(col("l_receiptdate"))) \
       .withColumn("l_commitdate", to_date(col("l_commitdate"))) \
       .withColumn("o_orderdate", to_date(col("o_orderdate")))

# label: is_late = receiptdate > commitdate
df = df.withColumn("is_late", (col("l_receiptdate") > col("l_commitdate")).cast("integer"))

# ============ Feature engineering (ONLY features known before receipt) ============
# planned lead time in days: commitdate - orderdate (planning horizon)
df = df.withColumn("planned_lead_days", datediff(col("l_commitdate"), col("o_orderdate")))

# numeric features: quantity, discount, extended price, order total (all known at order time)
# categorical features to index: shipmode, orderpriority
# drop rows with nulls in critical fields
df = df.dropna(subset=["o_orderdate", "l_commitdate", "l_receiptdate", "planned_lead_days", "l_quantity", "l_discount"])

# ============ Class balance diagnostics ============
counts = df.groupBy("is_late").count().collect()
print("Class distribution (is_late):", counts)

# ============ Index categorical features ============
indexer_ship = StringIndexer(inputCol="l_shipmode", outputCol="shipmode_index", handleInvalid="keep")
indexer_prio = StringIndexer(inputCol="o_orderpriority", outputCol="priority_index", handleInvalid="keep")

# ============ Assemble feature vector ============
feature_cols = [
    "l_quantity",
    "l_discount",
    "l_extendedprice",
    "o_totalprice",
    "planned_lead_days",
    "shipmode_index",
    "priority_index"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

# ============ Train/test split ============
train, test = df.randomSplit([0.8, 0.2], seed=42)

# ============ Pipelines and models ============
# Logistic Regression pipeline
lr = LogisticRegression(featuresCol="features", labelCol="is_late", maxIter=50)
pipeline_lr = Pipeline(stages=[indexer_ship, indexer_prio, assembler, lr])

# RandomForest pipeline (increase trees moderately)
rf = RandomForestClassifier(featuresCol="features", labelCol="is_late", numTrees=20, maxDepth=10)
pipeline_rf = Pipeline(stages=[indexer_ship, indexer_prio, assembler, rf])

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

print("=========== Results Summary =======================================================")
print("Logistic Regression:")
print("Model: ", results_lr["model"])
print("AUC: ", results_lr["auc"])
print("Accuracy: ", results_lr["accuracy"])
print("F1: ", results_lr["f1"])
print("--------------------------------")
print("RandomForest:")
print("Model: ", results_rf["model"])
print("AUC: ", results_rf["auc"])
print("Accuracy: ", results_rf["accuracy"])
print("F1: ", results_rf["f1"])
print("====================================================================================")

# ============ End ============
spark.stop()
