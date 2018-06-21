#!/bin/bash
echo "Downloading labelfiles from FTP server..."
if [ ! -d "data" ]; then
  mkdir "data"
fi
cd "data"
wget -m -nH --no-passive-ftp --cut-dirs=3 "ftp://anonymous@ftp.sonah.xyz/files/hackathon/data"
echo "Done!"
