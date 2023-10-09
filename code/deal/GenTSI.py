import os
from code.dataset import get_dataset
from easydict import EasyDict as edict
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont
import PIL.ImageDraw as ImageDraw

def genTsiImg(text, sketch, image):
    ''' 
    Composite mixed image of Text, Sketch and Image
    text is “The tennis player is beginning to serve a ball.”
    sketch is image format
    image is image format
    '''
    # text generate img 
    # https://blog.csdn.net/m0_46653437/article/details/112046275
    src_img = Image.new("RGB", (sketch.width+image.width, 20*len(text)+max(sketch.height, image.height)), (255,255,255))
    draw = ImageDraw.Draw(src_img)
    font = ImageFont.truetype("./code/deal/Times.ttc", size=20) # 在mac系统/System/Library/Fonts 中找了个字体。
    for i, txt in enumerate(text):
        draw.text((5,20*(i+1)),txt, font=font, fill=(0, 0, 0))

    src_img.paste(sketch, (0, 100))
    src_img.paste(image, (sketch.width, 100))
    return src_img

def get_default_sfsd_config(dataset="BaseSFSDDataset", root_path=os.path.expanduser('~/datasets/SFSD')):
    config = edict()
    config.dataset_root_path = root_path
    config.dataset = dataset
    config.seed = 25    
    return config

def gen_TSI(config, save_path, split):
    print(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Dataset = get_dataset(config, split)
    for item in tqdm(Dataset):
        mix_img = genTsiImg(item[1], item[0], item[2])
        mix_img.save(os.path.join(save_path, item[3]+'.png'))

if __name__ == '__main__':
    # dataset = "BaseFScocoDataset"
    # root_path = root_path=os.path.expanduser('~/datasets/fscoco/')
    # save_path = './TSI_fscoco/'

    # dataset = "BaseSFSDDataset"
    # root_path = root_path=os.path.expanduser('~/datasets/SFSD/')
    # save_path = './TSI_SFSD/'

    dataset = "BaseSketchycocoDataset"
    root_path = root_path=os.path.expanduser('~/datasets/SketchyCOCO/')
    save_path = './TSI_Sketchycoco/'

    
    config = get_default_sfsd_config(dataset, root_path)
    gen_TSI(config, save_path, 'test')
