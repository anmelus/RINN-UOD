# setting up anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
chmod +x Anaconda3-2022.10-Linux-x86_64.sh
./Anaconda3-2022.10-Linux-x86_64.sh
export PATH="/home/${USER}/anaconda3/bin:$PATH"
conda init

# setting up code and data
git clone https://github.com/samiragarwala/RINN-UOD.git
cd RINN-UOD
conda env create -f environment.yml
conda activate cs229p
pip3 install --upgrade pip
pip3 install gdown
gdown 1fruy7638b3d09CCRqUaGtUoZ_T4_sKsE
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y

echo "Setup complete!"
