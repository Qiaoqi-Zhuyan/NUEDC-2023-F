#!/bin/bash

# 自启动程序
sleep 15
cd ~/Desktop/diansaiF
echo "nano" | sudo -S chmod 777 /dev/ttyACM0
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
python3.6 pred_serial.py


