# The BetaBase

conda activate Env

The BetaBase consists of the climbs (a.k.a problems), climbing sequence (a.k.a. beta), and associated data for all problems in the Tension Board database (as of Dec. 16 2024).

The stack for constructing the BetaBase:
- Insta-Down to download videos of the beta for each problem
- MediaPipe for pose estimation in each video
- xxx to extract the beta
- SQLite (?) for the final database

If you are only interested in the BetaBase, you only need to download this file:
xxx

If you would like to install the whole stack to construct a database for yourself, see the installation section

## TODO
figure out how to handle posts with multiple links
        - can download and process both, then assign uuid to the climb with the best matching holds
figure out how to track limbs
figure out how to id holds given limb data
figure out how to reorient id'd holds to face-on
figure out how to matched id holds with holds from that climb
figure out best way to make that into sequence data
train model to predict beta given climb info
train model to predict grade given beta and climb info

                Videos --> Sports2D --> poses
                poses --> beta
                NN(board) --> beta
                NN(beta) --> grade

BetaBase will contain only climbs (TB1&2) that have public beta videos, so all we have to do for this is:
  - Download them
  - Use a pose estimation model (MediaPipe) to get pose estimates
  - Process to get climb beta
  - BetaBase will contain all of the OG TB data for the relevant climbs, plus beta sequence and pose estimates

## Issues
  - Pose estimation seems unreliable in a lot of these climbs, particularly for the feet because they are often occluded
	we could use the concept of a box
	- get the person's bounding box from the video
	- compare that to the bounding box of the used holds in a given position
	- the box should be similar in size and location if that position is used
	- person's bounding box also tells us what holds could be used in that frame
	- rapid changes in box indicate a move
	- maybe use similar concept using center of mass

## Codes
insta downloader:
https://github.com/x404xx/Insta-Down

mediapipe:
https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md


some ML inspo:
https://arxiv.org/pdf/2102.01788
https://github.com/Thomas-debug-creator/ML_Climbing/blob/master/Documentation/Presentation_Semester_Project.pdf
https://github.com/Saibo-creator/Awesome-LLM-Constrained-Decoding

https://github.com/davidpagnon/Sports2D


## **Installation**

Insta-Down using `pip`

```
git clone https://github.com/x404xx/Insta-Down.git
cd Insta-Down
virtualenv env
env/scripts/activate
pip install -r requirements.txt
```

also need mediapipe and opencv-python
Do we need the toml?

## **Legal Disclaimer**

> This was made for educational purposes only, nobody which directly involved in this project is responsible for any damages caused. **_You are responsible for your actions._**
