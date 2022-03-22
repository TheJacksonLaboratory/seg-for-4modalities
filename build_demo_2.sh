conda init bash

conda create -y --name ssif-aim2-demo2 python==3.8.8

conda activate ssif-aim2-demo2

pip install ./dist/seg_for_4modalities-0.0.7-py3-none-any.whl
