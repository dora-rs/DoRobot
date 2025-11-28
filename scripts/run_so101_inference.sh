conda activate op

python operating_platform/core/inference.py \
    --robot.type=so101 \
    --inference.single_task="start and test so101 arm." \
    --inference.dataset.repo_id="/home/HwHiAiUser/act/test-dataset-1" \
    --policy.path="/home/HwHiAiUser/act/0906_act_so101_test/checkpoints/005000/pretrained_model"
