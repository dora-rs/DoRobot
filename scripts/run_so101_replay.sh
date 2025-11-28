conda activate op

python operating_platform/core/replay.py \
    --robot.type=so101 \
    --replay.repo_id="20250903/dev/so101-test" \
    --replay.episode=0

