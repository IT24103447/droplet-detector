"""Sample hydrotester integration using Enter as the Start and Stop commands."""
import argparse

from droplet_detector import FrameDetector, OutputPaths, VideoSettings, start_test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/droplet-seg.pt")
    parser.add_argument("--detector", choices=("yolo", "vision"), default="yolo")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--frame-rate", type=float)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--roi-shape", choices=("rectangle", "ellipse"), default="rectangle")
    parser.add_argument("--vision-interval", type=float, default=2.0, help="Seconds between vision samples")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    settings = VideoSettings(
        model_path=args.model,
        resolution=tuple(args.resolution) if args.resolution else None,
        frame_rate=args.frame_rate,
        roi=tuple(args.roi) if args.roi else None,
        confidence_threshold=args.confidence,
        match_distance_px=30 if args.detector == "vision" else 10,
        detector_backend=args.detector, roi_shape=args.roi_shape, vision_interval_s=args.vision_interval,
    )
    detector = FrameDetector(settings.model_path, settings.confidence_threshold) if args.detector == "yolo" else None
    input("Press Enter to start the test.")
    session = start_test(args.camera_id, outputs=OutputPaths.in_directory(args.output_dir),
                         settings=settings, prepared_detector=detector)
    try:
        input("Press Enter to stop the test and save the results.")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        result = session.stop_test()
    print(result.model_dump_json(indent=2))
    return int(result.status == "error")


if __name__ == "__main__":
    raise SystemExit(main())
