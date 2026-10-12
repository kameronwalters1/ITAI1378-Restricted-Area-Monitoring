# Demo Video Script

Target length: 3 to 5 minutes. Record your screen and your own narration. The provided `results/demo_reel_no_audio.mp4` contains the real model output but does not satisfy the narration requirement by itself.

## 0:00 to 0:30 Introduction

“My project is an AI powered restricted-area monitoring proof of concept. It uses a pretrained YOLO11n model to detect a person, ByteTrack to keep an ID across frames, and a polygon rule to decide whether the person entered the restricted zone. The red alert appears only after the tracked person stays in the zone for half a second.”

## 0:30 to 1:15 Code and inputs

Show the repository and briefly open `src/run_poc.py`.

“The input is a folder of normal phone videos. I recorded three hallway clips with the phone stationary. I did not train the model or use face recognition. The code keeps only the COCO person class. The blue polygon is stored as normalized coordinates so it scales with the video resolution.”

## 1:15 to 2:30 Real output

Play the demo reel or the three files in `results/annotated_videos/`.

“A green box is a detected person outside the zone. Orange means the bottom-center point is inside but has not met the dwell time. Red means the system confirmed and logged the event. The first clip contains one clear entry. The second clip is the difficult close-camera case. The third clip contains an exit and a later re-entry, so it correctly creates two events.”

## 2:30 to 3:20 Results and struggle case

Show `results/metrics.csv` and the close-camera alert image.

“Manual review found four real events and the program logged all four, which gives 100 percent event recall on this small set. The close-camera clip was the weakest case. YOLO detected the person in only 27 of 219 frames because most of the body was outside the image. The event still worked, but a faster pass could be missed. I observed no extra event in 41.7 seconds, which is not enough video to prove the original false-alert-per-hour target.”

## 3:20 to 4:00 Conclusion

“The main change from my Blueprint was using the pretrained model without fine-tuning. That let me finish and evaluate the working application. With more time, I would record at least one hour of negative footage, test more camera angles, and add a simple zone calibration tool. The system is a proof of concept, not a production security product.”

## Recording checklist

- Narrate in your own voice.
- Show real input and real annotated output.
- Include the close-camera struggle case.
- Keep the video between 3 and 5 minutes.
- Upload as YouTube Unlisted or Google Drive with “Anyone with the link can view.”
- Test the link in a private browser window.
- Paste the link into Canvas comments and replace the README placeholder.
