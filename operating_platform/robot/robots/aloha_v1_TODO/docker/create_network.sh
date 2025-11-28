# 创建隔离网络（子网 172.18.0.0/24，网关 172.18.0.1）
docker network create \
  --driver bridge \
  --subnet 172.6.0.0/24 \
  --gateway 172.6.0.1 \
  dorobot_net