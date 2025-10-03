#!/usr/bin/env python3
import argparse, os
from pyspark.sql import SparkSession, functions as F, Window
from delta.tables import DeltaTable

def get_spark(app="DedupMerge"):
    spark = (SparkSession.builder.appName(app)
             .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
             .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
             .getOrCreate())
    return spark

def deduplicate(df, keys, ts_col):
    w = Window.partitionBy(*keys).orderBy(F.col(ts_col).desc())
    return (df.withColumn("_rn", F.row_number().over(w))
              .filter(F.col("_rn")==1).drop("_rn"))

def write_delta(df, path):
    (df.write.format("delta").mode("overwrite").save(path))

def ensure_gold(spark, silver_df, gold_path, keys):
    if not os.path.exists(gold_path):
        (silver_df.limit(0)
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema","true")
            .save(gold_path))

def upsert_to_gold(spark, silver_df, gold_path, keys):
    gold = DeltaTable.forPath(spark, gold_path)
    cond = " AND ".join([f"g.{k} = s.{k}" for k in keys])
    (gold.alias("g").merge(silver_df.alias("s"), cond)
         .whenMatchedUpdateAll()
         .whenNotMatchedInsertAll()
         .execute())

def main(bronze_csv, silver_path, gold_path, keys, ts_col):
    spark = get_spark()
    df = (spark.read.option("header",True).csv(bronze_csv)
          .withColumn("ts", F.col("ts").cast("long"))
          .withColumn("event_id", F.col("event_id").cast("long"))
          .withColumn("store_id", F.col("store_id").cast("long"))
          .withColumn("speed", F.col("speed").cast("double"))
          .withColumn("accel", F.col("accel").cast("double"))
          .withColumn("rel_speed", F.col("rel_speed").cast("double"))
          .withColumn("rel_distance", F.col("rel_distance").cast("double"))
          .withColumn("occlusion_ct", F.col("occlusion_ct").cast("int"))
          .withColumn("near_miss", F.col("near_miss").cast("int"))
    )

    silver = deduplicate(df, keys, ts_col)
    write_delta(silver, silver_path)
    ensure_gold(spark, silver, gold_path, keys)
    upsert_to_gold(spark, silver, gold_path, keys)
    print(f"Silver written to {silver_path} and upserted into Gold {gold_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bronze_csv", required=True)
    ap.add_argument("--silver_path", required=True)
    ap.add_argument("--gold_path", required=True)
    ap.add_argument("--keys", required=True, help="comma-separated keys, e.g. store_id,event_id")
    ap.add_argument("--ts", required=True, help="timestamp column, e.g. ts")
    args = ap.parse_args()
    keys = [k.strip() for k in args.keys.split(",")]
    main(args.bronze_csv, args.silver_path, args.gold_path, keys, args.ts)
