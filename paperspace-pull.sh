# copy visible files
rsync -rvz -e "ssh -p 4321" --progress paperspace@$PS:/home/paperspace/Desktop/whisper/* ./

# copy dotfiles / hidden files
rsync -rvz -e "ssh -p 4321" --progress paperspace@$PS:/home/paperspace/Desktop/whisper/.[^.]* ./