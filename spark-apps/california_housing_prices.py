import math

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, OneVsRest, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# ============ Spark session ============
spark = (
    SparkSession.builder.appName("California Block Groups Price Prediction")
    .master("local[*]")
    .config("spark.driver.memory", "16g")
    .config("spark.executor.memory", "60g")
    .config("spark.driver.maxResultSize", "16g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)

# ============ Load dataset ============
print("Carregando dataset California Housing (Block Groups)...")
df = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .csv("california_housing.csv")
)

print(f"Dataset carregado: {df.count()} block groups")
print("Schema original:")
df.printSchema()

# ============ Feature Engineering ============
print("\n" + "=" * 70)
print("Criando features para análise de REGIÕES...")
print("=" * 70)

# 1. FEATURE: Densidade de cômodos por HOUSEHOLD
print("→ Criando features de densidade por household...")
df = (
    df.withColumn(
        "quartos_por_household",
        when(col("households") > 0, col("total_rooms") / col("households")).otherwise(
            0
        ),
    )
    .withColumn(
        "bedrooms_por_household",
        when(
            col("households") > 0, col("total_bedrooms") / col("households")
        ).otherwise(0),
    )
    .withColumn(
        "pessoas_por_household",
        when(col("households") > 0, col("population") / col("households")).otherwise(0),
    )
    .withColumn(
        "ratio_bedrooms_rooms",
        when(
            col("total_rooms") > 0, col("total_bedrooms") / col("total_rooms")
        ).otherwise(0),
    )
)

# 2. FEATURE: Densidade populacional
print("→ Criando features de densidade populacional...")
df = df.withColumn(
    "densidade_populacional",
    when(col("total_rooms") > 0, col("population") / col("total_rooms")).otherwise(0),
)

# 3. FEATURE: Distância do centro (Los Angeles como referência principal)
print("→ Criando feature: distancia_centro_LA...")
CENTRO_LA_LAT = 34.0522
CENTRO_LA_LON = -118.2437


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Raio da Terra em km
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


haversine_udf = udf(haversine_distance, DoubleType())

df = df.withColumn(
    "distancia_LA",
    haversine_udf(
        col("latitude"), col("longitude"), lit(CENTRO_LA_LAT), lit(CENTRO_LA_LON)
    ),
)

# 4. FEATURE: Categorias de desenvolvimento
print("→ Criando feature: categoria_desenvolvimento...")
df = df.withColumn(
    "categoria_desenvolvimento",
    when(col("housing_median_age") < 15, "novo")
    .when(col("housing_median_age") < 35, "estabelecido")
    .otherwise("maduro"),
)

# 5. FEATURE: Categorias de renda
print("→ Criando feature: categoria_renda...")
df = df.withColumn(
    "categoria_renda",
    when(col("median_income") < 3, "baixa")
    .when(col("median_income") < 5, "media_baixa")
    .when(col("median_income") < 7, "media_alta")
    .otherwise("alta"),
)

# 6. FEATURE: Tamanho médio das households
df = df.withColumn(
    "tamanho_medio_household",
    when(col("households") > 0, col("population") / col("households")).otherwise(0),
)

# ============ Label Creation (Faixas de Valor Mediano) ============
print("→ Criando label: faixa_valor...")
df = df.withColumn(
    "faixa_valor",
    when(col("median_house_value") < 100000, "0_ate_100k")
    .when(col("median_house_value") < 150000, "1_100k_a_150k")
    .when(col("median_house_value") < 200000, "2_150k_a_200k")
    .when(col("median_house_value") < 300000, "3_200k_a_300k")
    .when(col("median_house_value") < 400000, "4_300k_a_400k")
    .otherwise("5_acima_400k"),
)

# ============ Selecionar features finais ============
feature_columns = [
    "housing_median_age",  # Idade mediana das casas na região
    "total_rooms",  # Total de cômodos na região
    "total_bedrooms",  # Total de quartos na região
    "population",  # População total da região
    "households",  # Número de households na região
    "median_income",  # Renda mediana da região
    "quartos_por_household",  # Feature 1: Quartos por household
    "bedrooms_por_household",  # Feature 2: Bedrooms por household
    "pessoas_por_household",  # Feature 3: Pessoas por household
    "ratio_bedrooms_rooms",  # Feature 4: Proporção bedrooms/rooms
    "distancia_LA",  # Feature 5: Distância de Los Angeles
    "densidade_populacional",  # Feature 6: Densidade populacional
    "tamanho_medio_household",  # Feature 7: Tamanho médio dos households
]

# Features categóricas para indexar
categorical_columns = [
    "categoria_desenvolvimento",
    "categoria_renda",
    "ocean_proximity",
]

print(
    f"\nFeatures selecionadas ({len(feature_columns)} numéricas + {len(categorical_columns)} categóricas):"
)
for i, feat in enumerate(feature_columns, 1):
    print(f"  {i}. {feat}")

# Selecionar colunas relevantes e remover nulos
df_clean = df.select(
    feature_columns + categorical_columns + ["faixa_valor", "median_house_value"]
).dropna()

print(f"\nDataset após limpeza: {df_clean.count()} block groups")

# Mostrar distribuição das faixas de valor
print("\nDistribuição das faixas de valor mediano:")
df_clean.groupBy("faixa_valor").count().orderBy("faixa_valor").show()

# ============ ANALISAR DISTRIBUIÇÃO DAS CLASSES ============
print("\nDistribuição detalhada das classes:")
class_distribution = (
    df_clean.groupBy("faixa_valor").count().orderBy("faixa_valor").collect()
)
for row in class_distribution:
    print(
        f"  {row['faixa_valor']}: {row['count']} regiões ({row['count']/df_clean.count()*100:.1f}%)"
    )

df_clean.repartition(1).write.mode("overwrite").option("header", "true").csv(
    "output/california_households_ml_dataset"
)

# ============ Preparação para ML ============
print("\n" + "=" * 70)
print("Preparando dados para Machine Learning...")
print("=" * 70)

# Indexar labels categóricas
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
    for c in categorical_columns
]

# Indexar label principal
label_indexer = StringIndexer(
    inputCol="faixa_valor", outputCol="label", handleInvalid="skip"
)

# Todas as features (numéricas + categóricas indexadas)
all_feature_columns = feature_columns + [f"{c}_index" for c in categorical_columns]

# Assembler
assembler = VectorAssembler(
    inputCols=all_feature_columns, outputCol="features", handleInvalid="skip"
)

# ============ Train/test split ============
train, test = df_clean.randomSplit([0.8, 0.2], seed=42)
print(f"Conjunto de treino: {train.count()} regiões")
print(f"Conjunto de teste:  {test.count()} regiões")

print("\nDistribuição das classes no conjunto de TREINO:")
train.groupBy("faixa_valor").count().orderBy("faixa_valor").show()

# ============================================================
#  Modelos Escolhidos
#  1. Random Forest (multiclasse nativo)
#  2. OneVsRest + Gradient Boosted Trees (GBT é binário)
# ============================================================

# ----- 1) RANDOM FOREST -----
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=20,
    subsamplingRate=0.8,
    featureSubsetStrategy="all",
    seed=42,
)

# ----- 2) ONE-VS-REST + GBT -----
gbt_binary = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=60,
    maxDepth=6,
    stepSize=0.05,
    subsamplingRate=0.8,
    seed=42,
)

gbt = OneVsRest(classifier=gbt_binary, labelCol="label", featuresCol="features")

# ============ Pipelines ============
pipeline_stages = indexers + [label_indexer] + [assembler]

pipeline_rf = Pipeline(stages=pipeline_stages + [rf])
pipeline_gbt = Pipeline(stages=pipeline_stages + [gbt])

# ============ Train models ============
print("\n" + "=" * 70)
print("Treinando modelos...")
print("=" * 70)

print("→ Random Forest...")
rf_model = pipeline_rf.fit(train)
print("  ✓ Concluído")

print("→ GBT (One-vs-Rest)...")
gbt_model = pipeline_gbt.fit(train)
print("  ✓ Concluído")

# ============ Predictions ============
rf_predictions = rf_model.transform(test)
gbt_predictions = gbt_model.transform(test)

# ============ Evaluation ============
print("\n" + "=" * 70)
print("Avaliando modelos...")
print("=" * 70)

# Avaliadores multiclasse
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
)
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall"
)


def evaluate_model(preds, name):
    return {
        "model": name,
        "accuracy": evaluator_acc.evaluate(preds),
        "f1": evaluator_f1.evaluate(preds),
        "precision": evaluator_precision.evaluate(preds),
        "recall": evaluator_recall.evaluate(preds),
    }


results_rf = evaluate_model(rf_predictions, "Random Forest")
results_gbt = evaluate_model(gbt_predictions, "GBT (One-vs-Rest)")

# ============ Results ============
print("\n" + "=" * 70)
print("RESULTADOS FINAIS - Predição de Faixa de Valor por REGIÃO")
print("=" * 70)


def print_results(res):
    print(f"  Accuracy:  {res['accuracy']:.4f}")
    print(f"  F1-Score:  {res['f1']:.4f}")
    print(f"  Precision: {res['precision']:.4f}")
    print(f"  Recall:    {res['recall']:.4f}")


print("\nRandom Forest:")
print_results(results_rf)

print("\nGBT (One-vs-Rest):")
print_results(results_gbt)
