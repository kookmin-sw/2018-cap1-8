소스 코드를 보관하는 폴더 
소스 코드 빌드 시, 필요한 파일은 일반적으로 lib 폴더를 생성하여 보관한다.
src 하위 폴더는 아래 조건 하에서 자유롭게 관리한다. 
	-동일한 소스 코드를 여러 폴더로 중복 관리하지 않는다. 
	-소스 코드 버전별로 폴더를 생성하지 않는다. 
	-직접 작성한 소스 코드만이 유지되도록 한다. 

1. 시스템 구축
시스템 설치 과정 
버전: ubuntu 14.04.02 LTS
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git
sudo apt-get install gcc make linux-headers-$(uname -r) git-core
CSITOOL_KERNEL_TAG=csitool-$(uname -r | cut -d . -f 1-2)
git clone https://github.com/dhalperi/linux-80211n-csitool.git
cd linux-80211n-csitool
git checkout ${CSITOOL_KERNEL_TAG}
cd ..
git clone https://github.com/dhalperi/linux-80211n-csitool-supplementary.git
sudo apt-get install libpcap-dev
git clone https://github.com/dhalperi/lorcon-old.git
cd linux-80211n-csitool
make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi modules
sudo make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi INSTALL_MOD_DIR=updates modules_install
sudo depmod
cd ..
for file in /lib/firmware/iwlwifi-5000-*.ucode; do sudo mv $file $file.orig; done
sudo cp linux-80211n-csitool-supplementary/firmware/iwlwifi-5000-2.ucode.sigcomm2010 /lib/firmware/
sudo ln -s iwlwifi-5000-2.ucode.sigcomm2010 /lib/firmware/iwlwifi-5000-2.ucode
make -C linux-80211n-csitool-supplementary/netlink
cd lorcon-old
./configure
make
sudo make install
cd linux-80211n-csitool-supplementary/injection
Make

2. Injection mode(패킷을 보내는 방법)
setup_injecttest2.sh가 있는 폴더에서 sudo bash setup_injecttest2.sh
sudo ./ Link\ to\ random_packets (총 패킷량) (한번에 보낼 패킷량) 1 (딜레이)

3. Monitor mode(패킷을 받는 방법)
setup_monitor_csitest.sh가 있는 폴더에서 sudo bash setup_monitor_csitest.sh
Supplementary/netlink 에서 sudo ./log_to_file (파일명)

4. 실행 방법
- new train으로 학습을 시켜 모델 파일 생성.
- csi tool Matlab utility 파일이 예측 실행 파일과 같은 폴더에 존재하도록 위치.
- intergrationPred.py 파일으로 예측

 



