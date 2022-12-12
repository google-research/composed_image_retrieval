# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pycocotools.coco import COCO
from collections import defaultdict
import random
import pandas as pd
from PIL import Image
import json
import numpy as np
import os

coco = COCO(annotation_file='annotations/instances_val2017.json')
cat_ids = coco.getCatIds()
def convert_coco_json_to_csv(filename='./annotations/instances_val2017.json', root='./val2017'):    
    s = json.load(open(filename, 'r'))
    out_file = 'coco_eval.csv'
    mask_dir = root+"_masked"
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)    
    out = open(out_file, 'w')
    out.write('id,query_regions,query_class,classes\n')    
    all_ids = []
    dict_id2cat = {item['id']:item['name'] for item in s['categories']}    
    for im in s['images']:
        all_ids.append(im['id'])        
    all_ids_ann = []
    id2anns = defaultdict(list)
    for ann in s['annotations']:
        image_id = ann['image_id']
        all_ids_ann.append(image_id)        
        x1 = ann['bbox'][0]
        x2 = ann['bbox'][0] + ann['bbox'][2]
        y1 = ann['bbox'][1]
        y2 = ann['bbox'][1] + ann['bbox'][3]
        label = dict_id2cat[ann['category_id']]        
        tmp = [x1, y1, x2, y2, label, ann]
        id2anns[image_id].append(tmp)        
    # Give query regions + classes not included in the query as a hint to retrieve images.
    class_count = 0
    for id_img in id2anns.keys():
        anns = id2anns[id_img]
        label_set = {}
        for ann in anns:
            label_set[ann[-2]] = label_set.get(ann[-2], 0) + 1
        label_set = list(label_set.keys())
        class_count += len(label_set)
        output = "%012d.jpg," %id_img
        image = Image.open(os.path.join(root, "%012d.jpg" %id_img))
        image = np.array(image)
        width, height = image.shape[0], image.shape[1]
        area_img = width * height
        cand_query = []
        for cand in anns:
            x1, y1, x2, y2 = map(lambda x: float(x), cand[:-2])
            area = (x2-x1) * (y2-y1)
            if 0.05 < area < 0.5 * area_img:
                cand_query.append(cand)
        if len(cand_query) >= 1:
            query_regions = random.sample(cand_query, k=1)
            for region in query_regions:
                query_label = region[-2]
                ann_region = region[-1]
                
                id_img = ann_region['image_id']
                filename = coco.imgs[id_img]['file_name']                
                image = Image.open(os.path.join(root, filename))
                image = np.array(image)
                mask = coco.annToMask(ann_region)
                width, height = mask.shape
                mask = mask.reshape(width, height,1)                
                if len(image.shape) == 2:
                    image = image.reshape(width, height, 1)
                image_masked = image * mask + (1-mask) * 255
                try:
                    im = Image.fromarray(image_masked)
                except:
                    image_masked = np.squeeze(image_masked, axis=2)
                    im = Image.fromarray(image_masked)                    
                im.save(os.path.join(mask_dir, filename))
                
                label_set.remove(query_label)
                output += ";".join(map(lambda x: str(x), region[:-2]))
                output += " "
                output += ","
                output += query_label
                output += ","
                output += ";".join(label_set)
                output += "\n"
            out.write(output)
    out.close()
    # Sort file by image id
    s1 = pd.read_csv(out_file)
    s1.sort_values('id', inplace=True)
    s1.to_csv(out_file, index=False)
    
convert_coco_json_to_csv()
