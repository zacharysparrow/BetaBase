# The BetaBase

The BetaBase consists of the climbs (a.k.a problems), climbing sequence (a.k.a. beta), and associated data for all problems in the Tension Board database (as of Dec. 16 2024).

The stack for constructing the BetaBase:
- Insta-Down to download videos of the beta for each problem
- xxx for pose estimation in each video
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


insta downloader:
https://github.com/x404xx/Insta-Down

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

## **Legal Disclaimer**

> This was made for educational purposes only, nobody which directly involved in this project is responsible for any damages caused. **_You are responsible for your actions._**
