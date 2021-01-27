pip install gdown

# Download WFLW dataset
unzip_dir='./wflw/' # unzip directory
f_id=1dtFIHkMc9H-9NjbRvqSsbc0bzDFlkdia
f="wflw_images.zip"                                                              
echo 'Downloading' $f ' ...' && gdown --id f_id && unzip -q $f -d $unzip_dir && rm $f # download, unzip, remove


# Download Widerface train dataset in yolo format
unzip_dir='./widerface/' # unzip directory
f_id=1VYxoZetzbvLysGbUYbAMTF5FepXocjDj
f="widerface_train_yolo.zip"                                                              
echo 'Downloading' $f ' ...' && gdown --id f_id && unzip -q $f -d $unzip_dir && rm $f # download, unzip, remove

