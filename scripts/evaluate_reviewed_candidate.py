"""Compare two models on held-out, provisional AI-reviewed annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def overlap(a, b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda r: max(0,r[2]-r[0])*max(0,r[3]-r[1])
    return intersection / max(area(a)+area(b)-intersection,1e-9)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--training-annotations",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,default=Path("models/droplet-seg.pt"))
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--specimen-crop",action="store_true")
    args=parser.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    rows=json.loads(args.annotations.read_text())
    training=json.loads(args.training_annotations.read_text())
    for field in ("sha256","group"):
        assert not ({r[field] for r in rows} & {r[field] for r in training}), "Hold-out overlap"
    assert all(r["split"]=="test" and r["review_status"]=="ai_visually_reviewed_provisional" for r in rows)
    args.output.mkdir(parents=True,exist_ok=False)
    summary=dict(confidence=.25,match_iou=.5,images=len(rows),specimen_crop=args.specimen_crop,
                 caveat="Small selected sample with provisional AI labels; not an expert accuracy audit or full-video timing evaluation.")
    for name,path in (("baseline",args.baseline),("candidate",args.candidate)):
        model=YOLO(str(path))
        totals=dict(tp=0,fp=0,fn=0,negative_frames=0,negative_frames_with_detections=0)
        details=[]
        for row in rows:
            image=cv2.imread(row["image"])
            x0,y0,x1,y1=row["crop"]
            truth=[[x0+b[0]/500*(x1-x0), y0+b[1]/500*(y1-y0),
                    x0+b[2]/500*(x1-x0), y0+b[3]/500*(y1-y0)] for b in row["boxes_preview_xyxy"]]
            source=image[y0:y1,x0:x1] if args.specimen_crop else image
            prediction=model.predict(source,conf=.25,imgsz=640,verbose=False,device="cpu")[0]
            boxes=prediction.boxes.xyxy.tolist()
            if args.specimen_crop:
                boxes=[[b[0]+x0,b[1]+y0,b[2]+x0,b[3]+y0] for b in boxes]
            matched=set();tp=0
            for box in boxes:
                scores=[(overlap(box,b),i) for i,b in enumerate(truth) if i not in matched]
                if scores and max(scores)[0]>=.5:
                    matched.add(max(scores)[1]);tp+=1
            stats=dict(tp=tp,fp=len(boxes)-tp,fn=len(truth)-tp)
            for key,value in stats.items():totals[key]+=value
            if not truth:
                totals["negative_frames"]+=1
                totals["negative_frames_with_detections"]+=bool(boxes)
            details.append(dict(image=row["image"],source=row["source"],time_s=row["time_s"],
                                truth=truth,boxes=boxes,scores=prediction.boxes.conf.tolist(),**stats))
            preview=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            draw=ImageDraw.Draw(preview)
            for box in truth:draw.rectangle(tuple(box),outline="cyan",width=3)
            for box in boxes:draw.rectangle(tuple(box),outline="red",width=2)
            preview.thumbnail((960,540))
            preview.save(args.output/f"{name}_{row['id']:02}.jpg")
        totals["precision"]=totals["tp"]/max(1,totals["tp"]+totals["fp"])
        totals["recall"]=totals["tp"]/max(1,totals["tp"]+totals["fn"])
        summary[name]=totals
        (args.output/f"{name}_predictions.json").write_text(json.dumps(details,indent=2))
    (args.output/"comparison.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
