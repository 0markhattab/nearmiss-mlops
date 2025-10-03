#!/usr/bin/env python3
import argparse, random, csv
from datetime import datetime, timedelta
random.seed(7)

def main(out):
    headers = ["event_id","store_id","ts","speed","accel","rel_speed","rel_distance","occlusion_ct","near_miss"]
    start = datetime(2024,1,1,8,0,0)
    rows = []
    eid = 1000
    for store in range(100, 120):
        t = start
        for _ in range(0, 300):
            # simulate telemetry
            speed = max(0, random.gauss(6, 2))  # m/s
            accel = random.gauss(0, 0.6)
            rel_speed = random.gauss(0, 3)
            rel_distance = max(0.5, abs(random.gauss(5, 2)))
            occlusion = max(0, int(random.gauss(1.2, 1)))
            risk = (max(0, -rel_speed) / (rel_distance+1e-3)) + (occlusion*0.25) + max(0, -accel*0.2)
            near = 1 if risk > 0.8 and rel_distance < 4 else 0
            row = [eid, store, int(t.timestamp()), round(speed,3), round(accel,3),
                   round(rel_speed,3), round(rel_distance,3), int(occlusion), near]
            rows.append(row)
            # occasional duplicates with later timestamp (to test dedup)
            if random.random() < 0.05:
                t_dup = t + timedelta(seconds=random.randint(1,30))
                row_dup = row.copy()
                row_dup[2] = int(t_dup.timestamp())
                rows.append(row_dup)
            eid += 1
            t += timedelta(seconds=5)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.out)
