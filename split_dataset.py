import os,shutil
import numpy as np
from lib.dataset.sample import SampleDataset
from lib.config import sample_cfg as cfg
def split_dataset():
    np.random.seed(42)
    img_root = "/home/user01/datasets/traffic/Sample_conbined"
    data = os.listdir(img_root)
    shuffled_index = np.random.permutation(len(data))
    rate = 0.3
    train_ind = shuffled_index[:-int(len(data) * rate)]
    test_ind = shuffled_index[-int(len(data) * rate):]
    train = "/home/user01/datasets/traffic/Sample_conbined/train"
    os.mkdir(train)
    val = "/home/user01/datasets/traffic/Sample_conbined/val"
    os.mkdir(val)
    for i in train_ind:
        src = os.path.join(img_root, data[i])
        dst = os.path.join(train, data[i])
        shutil.move(src, dst)
    for i in test_ind:
        src = os.path.join(img_root, data[i])
        dst = os.path.join(val, data[i])
        shutil.move(src, dst)

def test_py():
    sample= SampleDataset(cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE)
    print(sample[0])


if __name__ ==  "__main__":
    split_dataset()
    # test_py()