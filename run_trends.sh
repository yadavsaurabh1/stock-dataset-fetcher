#!/bin/zsh
# Runs trends_builder.py once any price build or other trends run has finished
# (the ranking needs the price files; two trends runs would share Google's rate limit).
# Safe to stop and re-run: finished work is skipped.
cd "$(dirname "$0")"
LOG=~/stock_dataset/trends.log

while pgrep -f "dataset_builder.py|trends_builder.py" > /dev/null; do
    echo "$(date '+%H:%M') waiting for another run to finish" >> $LOG
    sleep 600
done

caffeinate -i .venv/bin/python -u trends_builder.py "$@" >> $LOG 2>&1
echo "$(date '+%H:%M') trends run ended with exit code $?" >> $LOG
