pip install gdown

# Download pretrained weights
unzip_dir='./' # unzip directory
f_id=1_gVszd6i7LtiaTTiOj_zef91Qz-ehGDE
f="pretrained_weights.zip"                                                              
echo 'Downloading' $f ' ...' && gdown --id f_id && unzip -q $f -d $unzip_dir && mv 'pretrained_weights' 'weights' && rm $f # download, unzip, remove
