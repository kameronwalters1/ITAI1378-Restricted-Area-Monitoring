# AI Usage Log

**Student:** Kameron Walters  
**Course:** ITAI 1378 Computer Vision and AI  
**Project:** AI Powered Restricted Area Monitoring

I used OpenAI Codex and ChatGPT as an assistant. I reviewed the decisions and kept the final project limited to a pipeline I can explain. The entries below describe the major interactions rather than every short question.

| Date | Tool and request | Suggestion received | What I kept or changed | What I learned |
|---|---|---|---|---|

| 2026-10-11 | Codex, turn the idea into a feasible Blueprint | Use one camera, one person class, one polygon, and local alerts | Kept the narrow scope and deferred the dashboard | A smaller end-to-end system is easier to test honestly than a large unfinished design |
| 2026-10-11 | Codex, interpret the Final Build instructions | Kept the rubric structure and changed the tier description to Tier 2 | Detection plus tracking and decision logic counts as multiple working components |
| 2026-10-11 | Codex, prepare the YOLO/OpenCV environment | Install Ultralytics and OpenCV in an isolated project environment | Kept the isolated environment and pinned package versions | A requirements file makes the project easier to reproduce on another computer |
| 2026-10-11 | Codex, define the actual event rule | Track the person, test the bounding-box bottom center against a polygon, and require 0.5 seconds inside | Kept the rule and a 0.35 confidence threshold | A dwell rule prevents a single unstable frame from immediately creating an alert |
| 2026-10-11 | Codex, run and summarize the three hallway videos | Save annotated videos, alert frames, event rows, and timing data | Kept all measured outputs and manually checked the four alert frames | Detector metrics and application event metrics answer different questions |
| 2026-10-11 | Codex, compare results with the Blueprint | Report 4 of 4 events, disclose that mAP was not measured, and avoid claiming an hourly false-alert rate from 41.7 seconds | Kept the measured result and the limitation | A metric should be marked not measured when the required labels or test duration do not exist |
| 2026-10-11 | Codex, prepare the slides | Follow the required slide order and add the close-camera struggle case | Kept the structure and added speaker notes and a recording script | Showing a weakness with an explanation demonstrates engineering judgment |

## Verification before submission

- I can explain what each stage of `src/run_poc.py` does.
- I can explain why the polygon uses the bottom center of a person box.
- I can explain how the confidence and dwell thresholds affect alerts.
- I can explain why 41.7 seconds cannot prove a false-alert rate per hour.
- I will not use AI during the presentation.
