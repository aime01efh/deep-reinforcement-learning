#!/bin/bash
tensorboard --logdir=./log/ --host=127.0.0.1 --port=3000 &
echo "Wait around 10 seconds and open this link at your browser (ignore other outputs):"
echo "https://$WORKSPACEID-3000.$WORKSPACEDOMAIN"
