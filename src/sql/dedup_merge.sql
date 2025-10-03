-- Databricks SQL: window-based dedup then idempotent MERGE INTO Gold
WITH dedup AS (
  SELECT
    *, ROW_NUMBER() OVER (PARTITION BY store_id, event_id ORDER BY ts DESC) AS rn
  FROM silver.events_source  -- replace with your source table
)
MERGE INTO gold.events AS g
USING (SELECT * FROM dedup WHERE rn = 1) AS s
ON g.store_id = s.store_id AND g.event_id = s.event_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;
