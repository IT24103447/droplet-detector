"""Run a recorded hydrostatic test from the project terminal."""
import argparse

from droplet_detector import OutputPaths, VideoSettings, analyze_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the company recording")
    parser.add_argument("--output-dir", required=True, help="New folder for video, images and JSON")
    parser.add_argument("--model", default="models/droplet-seg.pt")
    parser.add_argument("--detector", choices=("yolo", "vision"), default="yolo")
    parser.add_argument("--test-start", type=float, default=0.0, help="Test start in seconds from clip start")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--roi-shape", choices=("rectangle", "ellipse"), default="rectangle")
    parser.add_argument("--vision-interval", type=float, default=2.0, help="Seconds between vision samples")
    parser.add_argument("--confirmation-hits", type=int, default=3)
    parser.add_argument("--match-distance", type=float)
    parser.add_argument("--candidate-gap", type=float, default=0.5)
    args = parser.parse_args()
    try:
        report = analyze_video(
            args.video,
            outputs=OutputPaths.in_directory(args.output_dir),
            settings=VideoSettings(
                model_path=args.model, roi=tuple(args.roi) if args.roi else None,
                confidence_threshold=args.confidence, confirmation_hits=args.confirmation_hits,
                match_distance_px=(args.match_distance if args.match_distance is not None
                                   else (30 if args.detector == "vision" else 10)),
                max_candidate_gap_s=args.candidate_gap,
                detector_backend=args.detector, roi_shape=args.roi_shape, vision_interval_s=args.vision_interval,
            ),
            test_start_s=args.test_start,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(report.model_dump_json(indent=2))
    return int(report.status == "error")


if __name__ == "__main__":
    raise SystemExit(main())
