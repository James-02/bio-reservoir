"""Inspect raw MIT-BIH records and the derived Kaggle heartbeat CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.mitbih_raw import (
    BeatWindowConfig,
    DEFAULT_KAGGLE_DIR,
    DEFAULT_MITDB_DIR,
    download_mitdb,
    extract_beat_windows,
    list_records,
    plot_class_templates,
    plot_record_segment,
    summarize_kaggle_heartbeat_dataset,
    summarize_record,
    summarize_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mitdb-dir", default=str(DEFAULT_MITDB_DIR))
    parser.add_argument("--kaggle-dir", default=str(DEFAULT_KAGGLE_DIR))
    parser.add_argument("--output-dir", default="results/raw_ecg_exploration")
    parser.add_argument("--download-mitdb", action="store_true")
    parser.add_argument("--all-records", action="store_true")
    parser.add_argument("--records", nargs="+", default=["100", "106", "114"])
    parser.add_argument("--plot-record", default=None, metavar="RECORD")
    parser.add_argument("--plot-beats", default=None, metavar="RECORD")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--lead", default="0")
    parser.add_argument("--pre-samples", type=int, default=90)
    parser.add_argument("--post-samples", type=int, default=96)
    return parser.parse_args()


def _resolve_records(args: argparse.Namespace) -> list[str]:
    if args.all_records:
        return list_records(args.mitdb_dir)
    return [str(record) for record in args.records]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download_mitdb:
        target_records = None if args.all_records else [str(record) for record in args.records]
        download_mitdb(args.mitdb_dir, records=target_records)

    records = _resolve_records(args)
    mitdb_summary = summarize_records(records, data_dir=args.mitdb_dir)
    kaggle_summary = summarize_kaggle_heartbeat_dataset(args.kaggle_dir)

    summary_payload = {
        "mitdb_records": [summarize_record(record, data_dir=args.mitdb_dir) for record in records],
        "kaggle": kaggle_summary,
    }

    (output_dir / "mitdb_summary.csv").write_text(mitdb_summary.to_csv(index=False))
    (output_dir / "source_summary.json").write_text(json.dumps(summary_payload, indent=2))

    lead = int(args.lead) if str(args.lead).isdigit() else args.lead
    if args.plot_record:
        plot_record_segment(
            args.plot_record,
            data_dir=args.mitdb_dir,
            output_path=output_dir / f"record-{args.plot_record}-segment.png",
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
        )

    if args.plot_beats:
        windows, metadata = extract_beat_windows(
            args.plot_beats,
            data_dir=args.mitdb_dir,
            config=BeatWindowConfig(
                pre_samples=args.pre_samples,
                post_samples=args.post_samples,
                lead=lead,
            ),
        )
        plot_class_templates(
            windows,
            metadata,
            output_path=output_dir / f"record-{args.plot_beats}-beat-templates.png",
        )

    print(f"Wrote exploration outputs to {output_dir}")


if __name__ == "__main__":
    main()