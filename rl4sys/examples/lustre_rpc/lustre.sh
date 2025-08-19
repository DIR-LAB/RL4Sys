#!/bin/bash

# ====== 固定变量（按你提供的信息）======
USER="GiriYomi"
SERVER="c220g5-110430.wisc.cloudlab.us"   # server0
CLIENTS=(                                  # node0, node1, node2 全做客户端
  "c220g5-110420.wisc.cloudlab.us"
  "c220g5-110419.wisc.cloudlab.us"
  "c220g5-120121.wisc.cloudlab.us"
)
LNET_IFACE="ens1f0"
FSNAME="hasanfs"
MGS_NID="10.10.1.1@tcp"                    # server0 ens1f0 的 IP


# ====== 7) 多客户端 MPI 连通性验证（按原版思路，3 节点）======
# hostfile：node0/node1/node2 各给 20 slots（示例，与原版一致）
for host in "${CLIENTS[@]}"; do
ssh -tt -p 22 ${USER}@${host} << 'EOF'
sudo su -
echo "node0 slots=20
node1 slots=20
node2 slots=20" > /root/hfile
# 将 hostnames 替换为实际可解析主机（此处沿用原版写法，mpirun 按 hostfile 条数 -np）
mpirun --hostfile /root/hfile --map-by node -np `cat /root/hfile|wc -l` hostname
exit
exit
EOF
done

echo "[*] All done. You can verify from server:"
echo "  ssh ${USER}@${SERVER} 'sudo lctl dl; sudo lfs mdts; sudo lfs osts; sudo lfs df -h'"
echo "  and on a client: 'df -h | grep ${FSNAME}; ls -l /mnt/${FSNAME}'"