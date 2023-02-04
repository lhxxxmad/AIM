#!/usr/bin/env bash

bash tools/dist_train.sh configs/recognition/vit/vitclip_base_diving48.py 4 --test-last --validate \
--cfg-options model.backbone.pretrained=openaiclip work_dir=work_dirs_vit/diving48/debug