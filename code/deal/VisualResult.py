import json
import os

from PIL import Image, ImageDraw, ImageFont
import PIL.ImageDraw as ImageDraw

def genComImg(sketch, ground_truth, retrival_images):
    ''' 
    Composite mixed image of Sketch and all Images
    '''
    nums_image = len(retrival_images)+1
    src_img = Image.new("RGB", (sketch.width*2, sketch.height*(nums_image//2+1)))
    src_img.paste(sketch, (0, 0))
    src_img.paste(ground_truth, (sketch.width, 0))
    for i in range(0, nums_image, 2):
        src_img.paste(retrival_images[i], (0, sketch.height*(i+1)))
        # src_img.paste(retrival_images[i+1], (sketch.width, sketch.height*(i+1)))
    return src_img

def readJson(path):
    with open(path, "r") as f:
        try:
            load_dict = json.load(f)
            return  load_dict
        except json.decoder.JSONDecodeError:
            print(path)
    return None

def visual(sketch_path, image_path, retrival_result, save_path=os.path.expanduser("./result/")):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for sketch_id in retrival_result.keys():
        if sketch_id not in retrival_result[sketch_id]:
            sketch = Image.open(os.path.join(sketch_path, sketch_id+".png"))
            ground = Image.open(os.path.join(image_path, sketch_id+".jpg"))
            images = []
            for img_id in retrival_result[sketch_id][:1]:
                images.append(Image.open(os.path.join(image_path, img_id+".jpg")))
            img = genComImg(sketch, ground, images)
            img.save(os.path.join(save_path, sketch_id+'.png'))

if __name__ == '__main__':
    sketch_path = os.path.expanduser("~/datasets/SFSD-open/sketchImgs/")
    image_path = os.path.expanduser("~/datasets/SFSD-open/images/")
    retrival_result_path = os.path.expanduser("./runs/retrival_result.json")
    retrival_result = readJson(retrival_result_path)
    visual(sketch_path, image_path, retrival_result)