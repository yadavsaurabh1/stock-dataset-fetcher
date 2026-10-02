#!/bin/zsh
# Runs hf_upload.py until it finishes, restarting it when it exits early (the watchdog
# exits when an upload hangs; a restart skips files already on Hugging Face).
# Arguments are passed through, e.g.: ./run_hf_upload.sh --public --markets US
cd "$(dirname "$0")"
LOG=~/stock_dataset/hf_upload.log

for attempt in {1..20}; do
    echo "$(date '+%H:%M') upload attempt $attempt" >> $LOG
    caffeinate -i .venv/bin/python -u hf_upload.py "$@" >> $LOG 2>&1 && exit 0
    sleep 60
done
echo "$(date '+%H:%M') giving up after 20 attempts" >> $LOG
exit 1
