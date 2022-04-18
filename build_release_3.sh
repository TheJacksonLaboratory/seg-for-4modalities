conda init bash

conda create -y --name ssif-aim2-v1.0.0 python==3.8.8

conda activate ssif-aim2-v1.0.0

pip install ./dist/seg_for_4modalities-1.0.0-py3-none-any.whl
