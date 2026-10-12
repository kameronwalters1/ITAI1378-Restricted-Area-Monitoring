# AI Powered Restricted Area Monitoring

**Course:** ITAI 1378 Computer Vision and AI  
**Student:** Kameron Walters  
**Final project tier:** Tier 2 Advanced  
**Status:** Working proof of concept tested on three self-recorded hallway videos

## Problem

A fixed camera can record a restricted hallway, but a person still has to notice when someone crosses the boundary. This proof of concept detects people, tracks them across frames, checks whether the bottom center of each bounding box enters a configured polygon, and saves a local alert after the person remains inside for 0.5 seconds.

## What I built

The application processes a video from start to finish:

1. OpenCV reads each frame.
2. A pretrained Ultralytics YOLO11n model detects the COCO `person` class.
3. ByteTrack assigns an ID across frames.
4. A point-in-polygon test checks the tracked person's floor position against the restricted zone.
5. A 0.5 second dwell rule confirms the event.
6. The program writes an annotated video, an alert image, and a CSV event row.

This is Tier 2 because person detection feeds a separate tracking and zone-decision component.

## Demo

**Narrated demo link for Canvas: ADD YOUTUBE OR GOOGLE DRIVE LINK HERE AFTER UPLOAD**

The annotated footage and alert images stay outside this public repository because they show a person and a private home interior. A separate private demo bundle contains the real model output. Use [`docs/demo_script.md`](docs/demo_script.md) to record the required 3 to 5 minute narrated version, upload it as Unlisted on YouTube or to Google Drive with link access, then replace the placeholder above.

## Data used

I recorded three portrait videos with a stationary phone in one home hallway. No public dataset or custom training set was used in the final build.

| Clip | Frames | Duration | Test condition | Confirmed events |
|---|---:|---:|---|---:|
| IMG_5462.mov | 487 | 16.2 s | Clear entry | 1 |
| IMG_5463.mov | 219 | 7.3 s | Person very close to the camera and partly outside the frame | 1 |
| IMG_5464.mov | 544 | 18.1 s | Exit followed by re-entry from a side doorway | 2 |
| **Total** | **1,250** | **41.7 s** | Three self-recorded clips | **4** |

The raw `.mov` files are intentionally excluded from the public repository. They remain local project inputs.

## Measured results

Manual review identified four real zone-entry events. The program logged all four and did not create an extra event in these clips.

| Blueprint measure | Target | Final result | Interpretation |
|---|---:|---:|---|
| Person mAP50 | Above 0.70 | Not measured | The final dataset has event labels, not frame-level bounding-box labels |
| Intrusion-event recall | Above 90% | 100% (4/4) | Met on this small test set |
| False alerts | Below 1 per hour | 0 observed in 41.7 s | Too little negative video to validate an hourly rate |
| Event latency | Below 500 ms | Worst clip p95: 18.9 ms per processed frame | Met on the test computer |

The weighted mean end-to-end processing time was 19.9 ms per frame, including model startup overhead. The model's median inference time was about 16.3 ms. These timings describe this computer and these videos only.

## Honest failure and struggle analysis

The close-camera clip was the hardest case. The person occupied only part of the right edge of the frame, so YOLO returned a person detection in 27 of 219 frames, or 12.3%. The system still accumulated the required 0.5 seconds and logged the event, but a faster pass or shorter clip could have been missed. The most useful next test is a longer negative video with people walking near the polygon without entering.

## What changed from the Blueprint

- The Blueprint proposed sampling public data and possibly fine-tuning. The final build uses pretrained YOLO11n without training because the baseline already detected the staged person events.
- The planned 1,200 to 1,800 labeled images became 1,250 real video frames without bounding-box labels. This kept the project focused on the working application but prevented an honest mAP calculation.
- The project tier is now Tier 2 because YOLO detection, ByteTrack tracking, and zone logic work together.
- The dashboard and cloud notifications remain future work. The final output stays local.

## Run the project

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_poc.py --input input_videos --output results --model yolo11n.pt
```

Place test `.mov` or `.mp4` files in `input_videos/`. Ultralytics downloads `yolo11n.pt` the first time if the file is not present. The default settings use the person class, a 0.35 confidence threshold, a 0.5 second dwell time, and the polygon defined near the top of `src/run_poc.py`.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── run_poc.py
├── docs/
│   ├── AI_usage_log.md
│   ├── demo_script.md
│   └── presentation.pdf
└── results/
    ├── README.md
    ├── events.csv
    ├── metrics.csv
    └── run_summary.json
```

## Limitations

- Three short clips do not support claims about long-term false-alert performance.
- The polygon is calibrated for one fixed camera angle.
- The bottom center of a bounding box is only an estimate of floor position.
- Severe cropping or occlusion can interrupt detection and tracking.
- The program does not identify people and does not perform face recognition.

## Privacy note

The source footage, annotated videos, and alert images are private demonstration evidence. They are intentionally excluded from the public GitHub package. The presentation contains selected stills for course submission, and the narrated demo should be shared only through the link-access settings required by the instructor.

## AI use and academic integrity

AI assistance helped organize the code, interpret errors, draft documentation, and prepare the presentation. I remained responsible for recording the input videos, running the test, checking event frames, understanding the pipeline, and reporting limitations honestly. The detailed log is in [`docs/AI_usage_log.md`](docs/AI_usage_log.md).

## References

- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/)
- [OpenCV documentation](https://docs.opencv.org/)
