#!/bin/bash
if [ ! -d VOCdevkit ]; then
    wget -c http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    tar zxvf VOCtest_06-Nov-2007.tar
fi