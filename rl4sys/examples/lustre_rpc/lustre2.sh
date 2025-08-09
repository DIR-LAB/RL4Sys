# ========= 3) 配置 OSS（node0；OST index=0,1）=========
ssh -tt -p 22 ${USER}@${OSS} << EOF
sudo su -
set -e
echo "options lnet networks=tcp(${LNET_IFACE})" > /etc/modprobe.d/lustre.conf

# 两个 OST（会清空分区）
mkfs.lustre --fsname=${FSNAME} --ost --mgsnode=${MGS_NID} --index=0 --reformat /dev/sdb1
mkfs.lustre --fsname=${FSNAME} --ost --mgsnode=${MGS_NID} --index=1 --reformat /dev/sdb2

mkdir -p /mnt/ost0 /mnt/ost1
mount -t lustre /dev/sdb1 /mnt/ost0
mount -t lustre /dev/sdb2 /mnt/ost1
exit
exit
EOF

# ========= 4) 客户端：LNet 配置与挂载 =========
for host in "${CLIENTS[@]}"; do
ssh -tt -p 22 ${USER}@${host} << EOF
sudo su -
set -e
echo "options lnet networks=tcp(${LNET_IFACE})" > /etc/modprobe.d/lustre.conf
mkdir -p /mnt/${FSNAME}
mount -t lustre ${MGS_NID}:/${FSNAME} /mnt/${FSNAME}
exit
exit
EOF
done

# ========= 5) SSH 免密（保留原版做法；若不需要，可删除本段）=========
rm -f all.txt
for host in "${CLIENTS[@]}"; do
ssh -tt -p 22 ${USER}@${host} << 'EOF'
sudo su -
set -e
mkdir -p /root/keys
chown root:root /root/keys
[ -f /root/.ssh/id_rsa.pub ] || { mkdir -p /root/.ssh && ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa >/dev/null; }
cat /root/.ssh/id_rsa.pub >> /root/keys/pub_key.txt
exit
exit
EOF
ssh -tt -p 22 ${USER}@${host} cat /root/keys/pub_key.txt >> all.txt
done

for host in "${CLIENTS[@]}"; do
scp -P 22 ./all.txt ${USER}@${host}:/tmp/all.txt
ssh -tt -p 22 ${USER}@${host} << 'EOF'
sudo su -
set -e
mkdir -p /root/.ssh
cat /tmp/all.txt >> /root/.ssh/authorized_keys
rm -f /tmp/all.txt
rm -rf /root/keys
cat >>/root/.ssh/config <<__EOF
Host *
  StrictHostKeyChecking no
__EOF
chmod 0600 /root/.ssh/config
exit
exit
EOF
done
rm -f all.txt

echo "=========== All done. Next steps ==========="
echo "On server:  ssh ${USER}@${SERVER} 'sudo lctl dl; sudo lfs mdts; sudo lfs osts; sudo lfs df -h'"
echo "On OSS:     ssh ${USER}@${OSS} 'mount | grep lustre; sudo lctl dl'"
echo "On clients: ssh ${USER}@${CLIENTS[0]} 'df -h | grep ${FSNAME}; ls -l /mnt/${FSNAME}'"