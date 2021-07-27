cd datasets
# use curl for big-sized google drive files to redirect
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=16SKjOii3SeRoPB_QXxCw2cwYu039BYMp" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > datasets.zip
unzip datasets.zip && rm -rf datasets.zip