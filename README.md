# Cluster Spark + HDFS

## O que é

Cluster com:
- **HDFS**: 1 namenode + 2 datanodes
- **Spark**: 1 master + 2 workers

## Como executar

### 1. Subir o cluster
```bash
docker-compose up -d
docker ps
```

**Interfaces:**
- Spark: http://localhost:8080
- HDFS: http://localhost:9870

### 2. Colocar dados no HDFS
```bash
docker exec -it namenode bash
hdfs dfs -put /data/input/products.csv /data/input/
hdfs dfs -ls /data/input/
exit
```

### 3. Rodar o job Spark
```bash
docker exec -it spark-master bash
spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark-apps/data_cleaning.py
exit
```

### 4. Ver os resultados
```bash
docker exec -it namenode bash
hdfs dfs -ls /data/output/cleaned_products/full/
hdfs dfs -ls /data/output/cleaned_products/summary/
hdfs dfs -cat /data/output/cleaned_products/summary/part-*.csv
exit
```

## O que o script faz

O `data_cleaning.py` processa dados de produtos:

1. **Limpeza**: remove espaços, padroniza textos
2. **Correção**: trata valores nulos, ajusta tipos
3. **Enriquecimento**: cria colunas `InventoryValue` e `IsAvailable`
4. **Agregação**: gera estatísticas por categoria
5. **Saída**: salva dados limpos + resumo no HDFS

**Entrada**: `products.csv` (Name, Brand, Category, Price, Stock, Availability)  
**Saídas**: 
- `full/`: dados completos limpos
- `summary/`: estatísticas por categoria

## Parar o cluster
```bash
docker-compose down
```
