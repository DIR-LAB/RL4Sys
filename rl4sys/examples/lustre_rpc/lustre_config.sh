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

# ====== 0) 在 server0 上对 /dev/sdb 分区：sdb1..sdb4 ======
#  sdb1=20G (MGS)
#  sdb2=20G (MDT)
#  sdb3=20G (OST0)
#  sdb4=40G (OST1)
ssh -tt -p 22 ${USER}@${SERVER} << 'EOF'
sudo su -
set -e
DISK="/dev/sdb"

echo ">>> DANGEROUS: Creating new GPT and partitions on ${DISK} (WILL DESTROY DATA)"
parted -s "${DISK}" mklabel gpt
# 对齐到 MiB 边界
parted -s "${DISK}" mkpart primary 1MiB 20GiB     # sdb1 MGS  20G
parted -s "${DISK}" mkpart primary 20GiB 40GiB    # sdb2 MDT  20G
parted -s "${DISK}" mkpart primary 40GiB 60GiB    # sdb3 OST0 20G
parted -s "${DISK}" mkpart primary 60GiB 100GiB   # sdb4 OST1 40G
partprobe "${DISK}"
sleep 2
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT "${DISK}"
exit
exit
EOF


# ====== 1) 配置并格式化 MGS + MDS（在 server0）======
# *** Change: ssh address, network interface (LNet) ***
ssh -tt -p 22 ${USER}@${SERVER} << EOF
sudo su -
echo "options lnet networks=tcp(${LNET_IFACE})" > /etc/modprobe.d/lustre.conf
mkfs.lustre --mgs --reformat /dev/sdb1
mkfs.lustre --fsname=${FSNAME} --mgsnode=${MGS_NID} --mdt --index=0 --reformat /dev/sdb2
mkdir -p /mnt/mgt /mnt/mdt
mount -t lustre /dev/sdb1 /mnt/mgt
mount -t lustre /dev/sdb2 /mnt/mdt
exit
exit
EOF


# ====== 2) 在 server0 上配置 OSS（两个 OST：index=0 与 1）======
ssh -tt -p 22 ${USER}@${SERVER} << EOF
sudo su -
echo "options lnet networks=tcp(${LNET_IFACE})" > /etc/modprobe.d/lustre.conf
mkfs.lustre --fsname=${FSNAME} --ost --mgsnode=${MGS_NID} --index=0 --reformat /dev/sdb3
mkfs.lustre --fsname=${FSNAME} --ost --mgsnode=${MGS_NID} --index=1 --reformat /dev/sdb4
mkdir -p /mnt/ost0 /mnt/ost1
mount -t lustre /dev/sdb3 /mnt/ost0
mount -t lustre /dev/sdb4 /mnt/ost1
exit
exit
EOF


# ====== 3) 客户端：写 lnet、挂载 Lustre ======
# *** Change: ssh address, network interface (ens1f0) ***
for host in "${CLIENTS[@]}"; do
ssh -tt -p 22 ${USER}@${host} << EOF
sudo su -
echo "options lnet networks=tcp(${LNET_IFACE})" > /etc/modprobe.d/lustre.conf
mkdir -p /mnt/${FSNAME}
mount -t lustre ${MGS_NID}:/${FSNAME} /mnt/${FSNAME}
exit
exit
EOF
done


# ====== 4) SSH 指纹准备（完全按原版保留）======
# 在所有客户端（含 node0/node1/node2）写入 lustre.conf（已在上一步），此处保持原版“指纹检查”结构
for host in "${CLIENTS[@]}"; do
ssh -tt -p 22 ${USER}@${host} << EOF
sudo su -
echo "options lnet networks=tcp(${LNET_IFACE})" > /etc/modprobe.d/lustre.conf
exit
exit
EOF
done


# ====== 5) 免密登录：收集并分发 root 公钥（原版做法）======
# 收集
rm -f all.txt
for host in "${CLIENTS[@]}"; do
ssh -tt -p 22 ${USER}@${host} << 'EOF'
sudo su -
mkdir -p /root/keys
chown root:root /root/keys
# 若 /root/.ssh/id_rsa 不存在，可考虑 ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa
cat /root/.ssh/id_rsa.pub >> /root/keys/pub_key.txt
exit
exit
EOF
ssh -tt -p 22 ${USER}@${host} cat /root/keys/pub_key.txt >> all.txt
done
cat all.txt

# 分发
for host in "${CLIENTS[@]}"; do
scp -P 22 ./all.txt ${USER}@${host}:/tmp/all.txt
ssh -tt -p 22 ${USER}@${host} << 'EOF'
sudo su -
mkdir -p /root/.ssh
cat /tmp/all.txt >> /root/.ssh/authorized_keys
rm -rf /root/keys /tmp/all.txt
cat >>/root/.ssh/config <<\__EOF
Host *
  StrictHostKeyChecking no
__EOF
chmod 0600 /root/.ssh/config
exit
exit
EOF
done
rm -f all.txt




# ====== 7) 多客户端 MPI 连通性验证（按原版思路，3 节点）======
# hostfile：node0/node1/node2 各给 20 slots（示例，与原版一致）
for host in "${CLIENTS[@]}"; do
ssh -tt -p 22 ${USER}@${host} << 'EOF'
sudo su -
echo "node0 slots=20
node1 slots=20
node2 slots=20" > /root/hfile
# 将 hostnames 替换为实际可解析主机（此处沿用原版写法，mpirun 按 hostfile 条数 -np）
mpirun --allow-run-as-root --hostfile /root/hfile --map-by node -np `cat /root/hfile|wc -l` hostname || true
exit
exit
EOF
done

echo "[*] All done. You can verify from server:"
echo "  ssh ${USER}@${SERVER} 'sudo lctl dl; sudo lfs mdts; sudo lfs osts; sudo lfs df -h'"
echo "  and on a client: 'df -h | grep ${FSNAME}; ls -l /mnt/${FSNAME}'"
