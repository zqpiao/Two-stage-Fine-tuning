

## How to reproduce Mixing_Separating


Following the original implementation, it consists of 2 steps:

- **Step1: Mixing**:
   - use the public available pre-trained model on base datasets as model initialization and further fine tune the bbox head with mixing few shot finetuning datasets.

- **Step2: Separating**:
   - use the mixing model from step1 as model initialization and further fine tune the bbox head with few shot datasets.


### An example of VOC split1 1 shot setting with 8 gpus

```
# step1: Mixing
python ./tools/detection/train.py \
    configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning_seed0_mixing.py 8

# step2: Separating
python ./tools/detection/train.py \
    configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning_seed0_separating.py 8

```


