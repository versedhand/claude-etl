#!/bin/bash
# Launch 24 parallel workers to re-embed message_embeddings
# Run from: /mnt/d/obs/life-code/claude-etl

NUM_WORKERS=24
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"
source venv/bin/activate

echo "Launching $NUM_WORKERS workers..."
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    python3 reembed_messages_worker.py --worker $i --num-workers $NUM_WORKERS &
done

echo "All workers launched. Waiting..."
wait
echo "All workers complete!"
