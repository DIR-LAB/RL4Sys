# 从 110420 拉回
#rsync -avz --progress \
#  -e "ssh -i ~/.ssh/id_ed25519" \
#  GiriYomi@c220g5-110420.wisc.cloudlab.us:/users/GiriYomi/RL4Sys/ \
#  /home/yomi/0Projects/RL4Sys/

# 或者从 110419 拉回（两台机器在同一站点时，这个路径通常是同一个NFS后端）
rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_ed25519" \
  GiriYomi@c220g5-110419.wisc.cloudlab.us:/users/GiriYomi/RL4Sys/ \
  /home/yomi/0Projects/RL4Sys/