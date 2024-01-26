import json
import os
import argparse
# 将检索结果top10，通过网页可视化出来。
def genEveryRow(sketch_path, sketch_name, text, image_path, image_names, sketch_type, image_type):
    html_row = '''
        <div class="image-row">
            <div class="input-text">{}</div>
            {}
        </div>
        '''
    html_container = '''
        <div class="image-container">
            <div class="image-name">{}</div>
            <img class="{}" src="{}">
        </div>
        '''
    containers = ""
    sketch_container = html_container.format(sketch_name, "", os.path.join(sketch_path, "{}.{}".format(sketch_name, sketch_type)))
    gt_container = html_container.format(sketch_name, "", os.path.join(image_path, "{}.{}".format(sketch_name, image_type)))
    image_containers = ""
    for image_name in image_names:
        if sketch_name == image_name:
            image_container = html_container.format(image_name, "green-border", os.path.join(image_path, "{}.{}".format(image_name,image_type)))
        else:
            image_container = html_container.format(image_name, " ", os.path.join(image_path, "{}.{}".format(image_name,image_type)))
        image_containers += image_container

    containers = sketch_container + gt_container + image_containers
    ret = html_row.format(text, containers)
    return ret

def genHtml(dataset_name, sketch_path, image_path, results, captions, sketch_type, image_type):
    style = '''
         <style>
            body {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                flex-direction: column;
                align-items: center;
                padding: 20px;}
            .image-row {
                display: flex;
                justify-content: space-around;
                margin-bottom: 20px;}
            .image-container {
                position: relative;
                display: flex;
                flex-direction: column;
                align-items: center;
                margin: 3px;}
            .image-name {
                font-size: 12px;
                position: absolute;
                bottom: 0;
                background: rgba(255, 255, 255, 0.1);
                padding: 5px;}
            .input-text {
                width: 200px;
                height: 100px;
                margin-top: 5px;}
            .green-border {
                border: 5px solid #00FF00; /* 绿色边框，5像素宽度 */
                padding: 5px; /* 为了保留原始图像的空间，可以添加一些内边距 */
            }
            img {
                width: 100px;
                height: 100px;
                margin-top: 5px;}
        </style>
        '''
    TEMPLATE = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{}</title>
            {}
        </head>
        <body>
        {}
        </body>
        </html>
    '''
    sketch_ids = results.keys()
    body = ""
    for sketch_id in sketch_ids:
        row = genEveryRow(sketch_path, sketch_id, captions[sketch_id]["captions"][0], image_path, results[sketch_id], sketch_type, image_type)
        body += row
    ret = TEMPLATE.format(dataset_name, style, body)
    return ret

def read_json(path):
    with open(path, "r") as f:
        try:
            all_captions = json.load(f)
            return all_captions
        except json.decoder.JSONDecodeError:
            print("don't have "+ path)

def write(dataset_name ,content):
    with open("{}.{}".format(dataset_name,'html'), 'w') as file:
        file.write(content)

if __name__ == '__main__':
    # python code/deal/Result2html.py
    # python code/deal/Result2html.py --dataset_name FSCOCO --result_path ./results/FScocoDataset/retrieval.json --caption_path /data/zzm/datasets/fscoco/FScocoTest_cat.json
    # python code/deal/Result2html.py --dataset_name SketchyCOCO --result_path ./results/SketchycocoDataset/retrieval.json --caption_path /data/zzm/datasets/SketchyCOCO/valInTrain.json
    # python code/deal/Result2html.py --dataset_name SketchyCOCO-lf --result_path ./results/SketchycocoLFDataset/retrieval.json --caption_path /data/zzm/datasets/SketchyCOCO-lf/test.json
    parser=argparse.ArgumentParser("tranfer retrieval result to html")
    parser.add_argument('--dataset_name',
                        default='SFSD',
                        choices=['SFSD','FSCOCO','SketchyCOCO','SketchyCOCO-lf'],
                        help='the dataset name')
    parser.add_argument('--result_path',
                        default="./results/SFSDDataset/retrieval.json",
                        help='the path of retrieval result')
    parser.add_argument('--caption_path',
                        default="/data/zzm/datasets/SFSD-open/test.json",
                        help='the path of caption file')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    if dataset_name == "SFSD":
        sketch_type = "png"
        image_type = "jpg"
        sketch_path = "sketchImgs/"
        image_path = "images/"
    elif dataset_name == "FSCOCO":
        sketch_type = "jpg"
        image_type = "jpg"
        sketch_path = "raster_sketches"
        image_path = "images/"
    elif dataset_name == "SketchyCOCO":
        sketch_type = "png"
        image_type = "png"
        sketch_path = "Scene/Sketch/paper_version/valInTrain"
        image_path = "Scene/GT/valInTrain"
    elif dataset_name == "SketchyCOCO-lf":
        sketch_type = "png"
        image_type = "png"
        sketch_path = "test/sketch"
        image_path = "test/image"
    results = read_json(args.result_path)
    captions = read_json(args.caption_path)
    html = genHtml(dataset_name, sketch_path, image_path, results, captions, sketch_type, image_type)
    write(dataset_name, html)