#!/bin/bash

ROOT="wd_public_02"
NEW_ROOT="wilddash"
TRAIN_SPLIT_LIST="train_split.txt"
VAL_SPLIT_LIST="val_split.txt"

  
# unzip file
mkdir $ROOT
mkdir $NEW_ROOT
unzip "${ROOT}.zip" -d $ROOT



# create dir structure
cd $NEW_ROOT
mkdir train
cd train
mkdir images
mkdir labels
cd ..

mkdir val
cd val
mkdir images
mkdir labels
cd ..

cd ..

# copy train files 
while read line; do
  cp "${ROOT}/$line" "${NEW_ROOT}/train/${line}"
done <$TRAIN_SPLIT_LIST

# copy val files 
while read line; do
  cp "${ROOT}/$line" "${NEW_ROOT}/val/${line}"
done <$VAL_SPLIT_LIST


# zip the final 
zip -r wilddash.zip $NEW_ROOT
