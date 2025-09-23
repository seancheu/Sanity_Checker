"""
python qa_step7.py
  --detections_csv step6_out/detections.csv
  --features_csv step4_out/features.csv
  --conf_thresh 0.70
  --dominance_thresh 0.75
  --out_dir step7_out
"""


import argparse, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def load_csv(path):
    rows=[]
    with open(path,"r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def write_csv(path, header, rows):
    with open(path,"w",newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--detections_csv", required=True)   # from Step 6
    ap.add_argument("--features_csv", default=None)      # optional: Step 4
    ap.add_argument("--conf_thresh", type=float, default=0.70)
    ap.add_argument("--dominance_thresh", type=float, default=0.75)  # carrier: dominant class share
    ap.add_argument("--out_dir", default="step7_out")
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    det=load_csv(args.detections_csv)
    feats = load_csv(args.features_csv) if args.features_csv else None
    snr_map={}
    if feats:
        for r in feats:
            snr_map[int(r["clip_index"])]=float(r["snr_db"])

    # Global stats
    confs=[float(r["confidence"]) for r in det]
    low_conf=[r for r in det if float(r["confidence"])<args.conf_thresh]

    # Per-carrier aggregation
    by_carrier=defaultdict(list)
    for r in det:
        cid=r.get("carrier_index","")
        cid=-1 if cid=="" else int(cid)
        by_carrier[cid].append(r)

    # Carrier-level conflict analysis
    conflict_rows=[]
    clean_rows=[]
    review_rows=[]
    for cid, rows in sorted(by_carrier.items(), key=lambda kv: kv[0]):
        labels=[r["pred_name"] for r in rows]
        confs_c=[float(r["confidence"]) for r in rows]
        hist=Counter(labels)
        dom, dom_cnt=hist.most_common(1)[0]
        share=dom_cnt/len(rows)
        mean_conf_dom = float(np.mean([c for l,c in zip(labels,confs_c) if l==dom])) if dom_cnt>0 else 0.0

        # Use median freq/bw for context
        f0s=[float(r["f_center_hz_rel"]) for r in rows if r.get("f_center_hz_rel","")!=""]
        bws=[float(r["bw_hz_est"]) for r in rows if r.get("bw_hz_est","")!=""]
        f0_med = np.median(f0s) if f0s else ""
        bw_med = np.median(bws) if bws else ""

        # Decide status
        status="clean"
        reason=""
        if share < args.dominance_thresh:
            status="review"; reason=f"low_dominance:{share:.2f}"
        if np.mean(confs_c) < args.conf_thresh:
            status="review"; reason = (reason+"; " if reason else "")+f"low_conf_mean:{np.mean(confs_c):.2f}"

        row=[cid, f0_med, bw_med, len(rows), dom, f"{share:.2f}", f"{mean_conf_dom:.2f}", status, reason]
        clean_rows.append(row) if status=="clean" else review_rows.append(row)

        # If multiple classes present, log full breakdown
        if len(hist)>1:
            breakdown="; ".join([f"{k}:{v}" for k,v in sorted(hist.items(), key=lambda kv: -kv[1])])
            conflict_rows.append([cid, f0_med, bw_med, len(rows), breakdown, f"{share:.2f}", f"{mean_conf_dom:.2f}"])

    # Low-confidence clips list
    low_rows=[]
    for r in low_conf:
        i=int(r["clip_index"]); c=r["pred_name"]; s=float(r["confidence"])
        snr = snr_map.get(i,"")
        low_rows.append([i,c,f"{s:.2f}",snr, r.get("carrier_index",""), r.get("f_center_hz_rel",""), r.get("bw_hz_est","")])

    # Write CSVs
    write_csv(out/"carriers_clean.csv",
              ["carrier_index","f_center_hz_rel_med","bw_hz_med","clips","dominant_mod","dominant_share","dominant_conf_mean","status","reason"],
              clean_rows)
    write_csv(out/"carriers_review.csv",
              ["carrier_index","f_center_hz_rel_med","bw_hz_med","clips","dominant_mod","dominant_share","dominant_conf_mean","status","reason"],
              review_rows)
    write_csv(out/"carrier_conflicts.csv",
              ["carrier_index","f_center_hz_rel_med","bw_hz_med","clips","class_breakdown","dominant_share","dominant_conf_mean"],
              conflict_rows)
    write_csv(out/"clips_low_conf.csv",
              ["clip_index","pred","confidence","snr_db","carrier_index","f_center_hz_rel","bw_hz_est"],
              low_rows)

    # Quick text summary
    with open(out/"qa_summary.txt","w", encoding="utf-8") as f:
        f.write(f"Total clips: {len(det)}\n")
        f.write(f"Low-conf clips (<{args.conf_thresh:.2f}): {len(low_conf)}\n")
        f.write(f"Total carriers: {len(by_carrier)}\n")
        f.write(f"Carriers to review: {len(review_rows)}\n")
        f.write("Heuristics: low dominance or low mean confidence -> review.\n")
    print("[OK] Wrote QA outputs to", out)

if __name__=="__main__":
    main()
