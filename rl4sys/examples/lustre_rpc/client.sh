   # 1 GB write workload
   dd if=/dev/zero of=/mnt/hasanfs/testfile bs=4M count=64 oflag=direct
   # Mixed read/write with fio
   fio --name=lustre_test --directory=/mnt/hasanfs \
       --size=2G --rw=randrw --bs=1M --ioengine=libaio --numjobs=4 --runtime=120


    fio --name=read10m \
        --filename=/mnt/hasanfs/rpc_test/testfile \
        --rw=randread --bs=1M \
        --ioengine=libaio --direct=1 \
        --iodepth=32 \
        --time_based --runtime=20