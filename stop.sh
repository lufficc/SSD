ps -aux | grep 'python train.py' | awk '{print $2}' | xargs kill -9
