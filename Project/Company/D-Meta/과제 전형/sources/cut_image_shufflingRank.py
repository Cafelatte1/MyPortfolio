GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

import sys
import random as rnd
from itertools import permutations
from datetime import datetime

from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as tvf

import warnings
warnings.filterwarnings('ignore')

from helper_functions import *

class CFG:
    debug = False
    image_root_path = "./data/"
    # h, w
    target_grid = (None, None)
    operations = {
        "mirroring": lambda x: tvf.hflip(x),
        "flipping": lambda x: tvf.vflip(x),
        "rotation": lambda x: tvf.rotate(x, 90, expand=True)[:, :x.shape[2], :x.shape[1]],
    }
    inverse_operations = {
        "mirroring": lambda x: tvf.hflip(x),
        "flipping": lambda x: tvf.vflip(x),
        "rotation": lambda x: tvf.rotate(x, -90, expand=True)[:, :x.shape[2], :x.shape[1]],
    }
    shuffled_image_name_length = 4
    max_loss_value = 1000.0

# 이미지별 모든 연산 독립적으로 적용
def do_random_operation(image, operation_order):
    if CFG.debug:
        seed_everything()
    x = image.clone()
    for op in operation_order[rnd.randint(0, len(operation_order) - 1)]:
        if rnd.random() > 0.5:
            x = CFG.operations[op](x)
    return x

def main():
    # parsing arguments
    target_grid_h, target_grid_w = sys.argv[1:][1], sys.argv[1:][2]
    if (not target_grid_h.isdigit()) or (not target_grid_w.isdigit()):
        print("value error on the number of grid")
        return -1
    CFG.target_grid = (int(target_grid_h), int(target_grid_w))
    image_name = sys.argv[1:][0]
    prefix = sys.argv[1:][3]
    seed = int(sys.argv[1:][4]) if sys.argv[1:][4].isdigit() else int(datetime.now().timestamp())
    
    # loading raw image
    image_path = CFG.image_root_path + image_name
    shuffled_image_root_path = CFG.image_root_path + "random_" + image_name.split(".")[0] + "/"
    createFolder(shuffled_image_root_path)
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    c, h, w = image.shape

    # set grid & operation ordering list
    grid_h, grid_w = h // CFG.target_grid[0], w // CFG.target_grid[1]
    operation_order = list(permutations(CFG.operations.keys()))

    # remove images in target folder
    for fpath in os.listdir(shuffled_image_root_path):
        os.remove(shuffled_image_root_path + fpath)

    name_cnt = 0
    for idx_h in range(CFG.target_grid[0]):
        for idx_w in range(CFG.target_grid[1]):
            seed_everything(seed)
            cropped_image = image[:, (idx_h * grid_h):((idx_h+1) * grid_h), (idx_w * grid_w):((idx_w+1) * grid_w)]
            cropped_image = do_random_operation(cropped_image, operation_order)
            cropped_image = transforms.ToPILImage()(cropped_image)
            cropped_image.save(shuffled_image_root_path + f"{prefix}_{generate_random_name(CFG.shuffled_image_name_length, seed=seed)}.jpg")
            name_cnt += 1
            seed += 1

if __name__ == "__main__":
    main()
