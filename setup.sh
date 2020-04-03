sudo apt install -y python3-pip p7zip-full
pip3 install -r requirements.txt
wget -O Data/StackExpert.7z https://docs.google.com/uc?id=1u1iTWKbG2v6TvxCRQHvgOnzBC0ib0N5K&export=download
7za x Data/StackExpert.7z -oData