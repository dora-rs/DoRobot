python operating_platform/dataset/visual/visualize_dataset_html.py \
  --repo-id user/daoshui_zyy0627Copy_204

python operating_platform/dataset/visual/visualize_dataset_html.py \
  --repo-id 20250705/user/''$'\345\217\240\350\241\243\346\234\215''_yance_244'

python operating_platform/dataset/visual/visualize_dataset_html.py \
  --repo-id '20250831/dev/so101-test'


python operating_platform/dataset/visual/visualize_dataset_html.py \
  --repo-id 20250820/dev/''$'\345\200\222\346\260\264''_Pour the waterCopy0725testCopy_226'

python operating_platform/dataset/visual/visualize_dataset_html.py \
  --repo-id "20250822/user/倒水_Pour the waterCopy0725testCopy_226"

python operating_platform/dataset/visual/visualize_dataset_html.py \
  --repo-id "20250820/user/倒水_Pour the waterCopy0725testCopy_226"

python operating_platform/dataset/visual/visual_dataset.py \
    --repo-id 20250820/dev/''$'\345\200\222\346\260\264''_Pour the waterCopy0725testCopy_226' \
    --episode-index 0 \
    --mode distant

python operating_platform/dataset/visual/visual_dataset.py \
    --repo-id "20250822/dev/水果收纳_测试采集任务11CopyCopy_276" \
    --episode-index 4 \
    --mode distant

python operating_platform/dataset/visual/visual_dataset.py \
    --repo-id "20250903/dev/so101-test" \
    --episode-index 0 \
    --mode distant

python operating_platform/dataset/visual/visual_dataset.py \
    --repo-id "20250826/dev/倒水_111_277" \
    --episode-index 0 \
    --mode distant


python operating_platform/core/replay.py \
    --robot aloha \
    --dataset.repo_id "20250826/dev/倒水_111_277" \
    --dataset.episode 0