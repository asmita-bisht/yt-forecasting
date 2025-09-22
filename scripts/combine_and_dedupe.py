"""
conftest.py

This file provides reusable fixtures and utilities for all of the tests.

"""
import argparse, os, sys, glob, io, zipfile, json, datetime as dt
import pandas as pd


# file discovery & readers
def find_files(paths, exts={".csv", ".zip"}):
    files = []
    for root in paths:
        for ext in exts:
            files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(set(files))

# centralize options (dtype, encoding, engine)
def read_csv_robust(f, **kwargs):
    return pd.read_csv(f, **kwargs)

# yield a DataFrame for each CSV (plain or inside ZIP).
def iter_rows_from_path(path, encoding=None):
    
    if path.lower().endswith(".csv"):
        df = read_csv_robust(path, encoding=encoding)
        df["source_file"] = os.path.relpath(path)
        yield df
    elif path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    with z.open(name) as fh:
                        data = fh.read()
                    df = read_csv_robust(io.BytesIO(data), encoding=encoding)
                    df["source_file"] = f"{os.path.relpath(path)}::{name}"
                    yield df

# union columns, with frequently seen columns first, preserving order stability
def unify_columns(dfs):
    if not dfs:
        return pd.DataFrame()
    freq = {}
    for d in dfs:
        for c in d.columns:
            freq[c] = freq.get(c, 0) + 1
    ordered_cols = sorted(freq.keys(), key=lambda c: (-freq[c], c))
    return pd.concat([d.reindex(columns=ordered_cols) for d in dfs],
                     ignore_index=True, sort=False)

# schema normalization helpers
def _iso_duration_to_seconds(s):
    # minimal PT#H#M#S parser to seconds (returns None if unparsable)
    try:
        if not isinstance(s, str) or not s.startswith("PT"):
            return None
        total, num = 0, ""
        for ch in s[2:]:
            if ch.isdigit():
                num += ch
            else:
                if ch == "H" and num: total += int(num) * 3600
                if ch == "M" and num: total += int(num) * 60
                if ch == "S" and num: total += int(num)
                num = ""
        return total
    except Exception:
        return None
    
#rename common columns, synthesize missing basics, drop bulky noise, compute non-null count.
def normalize_schema(combined: pd.DataFrame) -> pd.DataFrame:
    df = combined.copy()

    # rename common synonyms to project schema
    rename_map = {
        "viewCount": "views_72h",
        "likeCount": "likes_72h",
        "commentCount": "comments_72h",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # build tags_str from raw list if needed
    if "tags_str" not in df.columns and "tags" in df.columns:
        df["tags_str"] = df["tags"].apply(
            lambda x: ",".join(x) if isinstance(x, (list, tuple)) else (x if isinstance(x, str) else None)
        )

    # ensure publishedAt_utc if only string publishedAt exists
    if "publishedAt_utc" not in df.columns and "publishedAt" in df.columns:
        df["publishedAt_utc"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")

    # ensure duration_sec if only ISO8601 duration exists
    if "duration_sec" not in df.columns and "duration" in df.columns:
        df["duration_sec"] = df["duration"].apply(_iso_duration_to_seconds)

    # drop bulky/noisy columns that bias non-null counts
    drop_like = [
        "thumbnails", "thumbnail", "localized", "topic", "liveStreamingDetails",
        "defaultLanguage", "defaultAudioLanguage", "contentRating",
        "madeForKids", "paidProductPlacement"
    ]
    to_drop = [c for c in df.columns if any(c.startswith(pfx) for pfx in drop_like)]
    if "tags_str" in df.columns and "tags" in df.columns:
        to_drop.append("tags")  # drop raw list if we have tags_str
    if to_drop:
        df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True, errors="ignore")

    # compute non-null count on a CORE whitelist only (prevents bias in dedupe)
    CORE_COLS = [
        "videoId","publishedAt_utc","duration_sec",
        "title","description","tags_str",
        "categoryId","channelId",
        "views_72h","likes_72h","comments_72h",
        "channel_viewCount","channel_subscriberCount","channel_videoCount",
        "snapshot_at_utc","approx_age_hours",
    ]
    present_core = [c for c in CORE_COLS if c in df.columns]
    df["_non_null_count"] = df[present_core].notna().sum(axis=1)

    return df


# dedupe logic
def dedupe_by_video(df: pd.DataFrame) -> pd.DataFrame:
    if "videoId" not in df.columns:
        raise SystemExit("No 'videoId' column found; cannot dedupe.")

    sort_by = ["_non_null_count"]; ascending = [False]  # most complete first
    if "pulled_at" in df.columns:
        # ensure parsed
        df["pulled_at"] = pd.to_datetime(df["pulled_at"], utc=True, errors="coerce")
        sort_by.append("pulled_at"); ascending.append(False)
    if "publishedAt_utc" in df.columns:
        sort_by.append("publishedAt_utc"); ascending.append(False)
    if "source_file" in df.columns:
        sort_by.append("source_file"); ascending.append(True)  # deterministic

    sdf = df.sort_values(sort_by, ascending=ascending, kind="mergesort")
    out = sdf.drop_duplicates(subset=["videoId"], keep="first").copy()
    out.drop(columns=["_non_null_count"], inplace=True, errors="ignore")
    if "publishedAt_utc" in out.columns:
        out = out.sort_values("publishedAt_utc").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


# CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Input dirs (e.g., data/interim data/raw)")
    ap.add_argument("--output", required=True,
                    help="Output Parquet path (e.g., data/processed/all_days.parquet)")
    ap.add_argument("--also-csv", action="store_true",
                    help="Also write CSV next to Parquet")
    ap.add_argument("--encoding", default=None,
                    help="CSV encoding override (e.g., utf-8, latin-1)")
    args = ap.parse_args()

    files = find_files(args.inputs, exts={".csv", ".zip"})
    if not files:
        raise SystemExit("No CSV/ZIP files found under: " + ", ".join(args.inputs))
    print(f"Found {len(files)} files")

    dfs = []; loaded = 0
    for p in files:
        try:
            for df in iter_rows_from_path(p, encoding=args.encoding):
                dfs.append(df); loaded += len(df)
        except Exception as e:
            print(f"Skipped {p}: {e}", file=sys.stderr)

    if not dfs:
        raise SystemExit("No rows loaded. Check encodings and folder paths.")

    print(f" Loaded {loaded:,} rows across {len(dfs)} tables")

    #  combine, normalize schema, filter Shorts, dedupe, save
    combined = unify_columns(dfs)
    started_rows = len(combined)
    print(f"Concatenated rows: {started_rows:,}")

    # normalize schema
    combined = normalize_schema(combined)

    #  filter out Shorts (duration_sec ≤ 180) when duration is known 
    shorts_removed = 0
    if "duration_sec" in combined.columns:
        dsec = pd.to_numeric(combined["duration_sec"], errors="coerce")
        mask_shorts = dsec.notna() & (dsec <= 180)
        shorts_removed = int(mask_shorts.sum())
        if shorts_removed > 0:
            combined = combined.loc[~mask_shorts].copy()
        print(f"Shorts removed (≤180s): {shorts_removed:,}")
    else:
        print("'duration_sec' not found; Shorts filter skipped.")

    # dedupe
    before = len(combined)  # after shorts filtering
    deduped = dedupe_by_video(combined)
    after = len(deduped)
    duplicates_removed = before - after

    # one terminal line with all key counts for reference
    print(
        f"[STATS] Started: {started_rows:,} | "
        f"Shorts removed: {shorts_removed:,} | "
        f"Duplicates removed: {duplicates_removed:,} | "
        f"Final rows: {after:,}"
    )

    # save outputs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    deduped.to_parquet(args.output, index=False)

    out_csv = None
    if args.also_csv:
        out_csv = args.output.rsplit(".", 1)[0] + ".csv"
        deduped.to_csv(out_csv, index=False)

    # summary JSON
    summary = {
        "generated_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": args.inputs,
        "files_scanned": len(files),
        "rows_loaded": int(loaded),
        "started_rows": int(started_rows),
        "shorts_removed": int(shorts_removed),
        "rows_after_concat": int(started_rows),     
        "rows_after_dedupe": int(after),
        "duplicates_removed": int(duplicates_removed),
        "output_parquet": os.path.abspath(args.output),
        "output_csv": os.path.abspath(out_csv) if out_csv else None,
    }
    summ_path = args.output.rsplit(".", 1)[0] + "_summary.json"
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
