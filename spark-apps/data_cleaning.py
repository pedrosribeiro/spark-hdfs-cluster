from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, length, round, trim, upper, when

# Inicializa a Spark Session
spark = SparkSession.builder.appName("ProductDataCleaning").getOrCreate()

# Caminhos no HDFS
input_path = "hdfs://namenode:9000/data/input/products.csv"
output_path = "hdfs://namenode:9000/data/output/cleaned_products/"

# Leitura do CSV
df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

# Etapa 1 — Limpeza básica
# Remove espaços extras e padroniza nomes de marca e categoria
df_clean = (
    df.withColumn("Name", trim(col("Name")))
    .withColumn("Brand", upper(trim(col("Brand"))))
    .withColumn("Category", upper(trim(col("Category"))))
)

# Etapa 2 — Tratamento de valores ausentes e correção de tipos
# Substitui nulos no estoque e preço por 0 e converte tipos
df_clean = (
    df_clean.withColumn("Stock", when(col("Stock").isNull(), 0).otherwise(col("Stock")))
    .withColumn("Price", when(col("Price").isNull(), 0).otherwise(col("Price")))
    .withColumn("Price", col("Price").cast("double"))
    .withColumn("Stock", col("Stock").cast("integer"))
)

# Etapa 3 — Criação de coluna derivada
# Valor total potencial de estoque = Price * Stock
df_clean = df_clean.withColumn("InventoryValue", round(col("Price") * col("Stock"), 2))

# Etapa 4 — Enriquecimento
# Marca como disponível = disponibilidade "in_stock" ou "pre_order"
df_clean = df_clean.withColumn(
    "IsAvailable",
    when(col("Availability").isin("in_stock", "pre_order"), True).otherwise(False),
)

# Etapa 5 — Análise resumida por categoria
df_summary = (
    df_clean.groupBy("Category")
    .agg(
        count("*").alias("TotalProducts"),
        round(avg("Price"), 2).alias("AvgPrice"),
        round(avg("Stock"), 2).alias("AvgStock"),
        round(avg("InventoryValue"), 2).alias("AvgInventoryValue"),
    )
    .orderBy(col("TotalProducts").desc())
)

# Salva resultados no HDFS
df_clean.repartition(1).write.mode("overwrite").option("header", "true").csv(
    output_path + "full/"
)
df_summary.repartition(1).write.mode("overwrite").option("header", "true").csv(
    output_path + "summary/"
)

spark.stop()
