cd results/visual
# use curl for big-sized google drive files to redirect
# infos_intention_changing_animation.p
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=1GXC6W0SipVo1kwabajoLqUhxcz-loC1k" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > infos_intention_changing_animation.p
# use wget for small-sized google drive files
# original_traj_intention_changing_animation.p
wget -O original_traj_intention_changing_animation.p "https://drive.google.com/uc?export=download&id=1JFZbWGcmaiSY6CK05E-JvBSDgQ2Tzb6p"
