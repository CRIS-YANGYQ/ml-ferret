import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
sys.path.append('/comp_robot/yangyuqin/workplace/Multi-model/models/ml-ferret')
# Added
from ferret.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from ferret.model.builder import load_pretrained_model
from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ferret.conversation import conv_templates, SeparatorStyle
from ferret.utils import disable_torch_init
from PIL import Image
import math
import pdb
import numpy as np
from copy import deepcopy
from functools import partial
import re
from torch.utils.data import Dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


"""
python3 -m ferret.model.apply_delta \
    --base /comp_robot/yangyuqin/workplace/Multi-model/models/ml-ferret/checkpoints/base/lmsys/vicuna-7b-v1.5 \
    --target /comp_robot/yangyuqin/workplace/Multi-model/models/ml-ferret/checkpoints/target/ferret-7b \
    --delta /comp_robot/yangyuqin/workplace/Multi-model/models/ml-ferret/checkpoints/delta/ferret-7b-delta



  "images": [
    {
      "license": 4,
      "file_name": "000000397133.jpg",
      "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
      "height": 427,
      "width": 640,
      "date_captured": "2013-11-14 17:02:52",
      "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
      "id": 397133
    }],

    "annotations": [
    {
      "area": 702.1057499999998,
      "iscrowd": 0,
      "image_id": 289343,
      "bbox": [
        473.07,
        395.93,
        38.65,
        28.67
      ],
      "category_id": 18,
      "id": 1768
    },..],

    "categories": [
    {
      "supercategory": "person",
      "id": 1,
      "name": "person"
    },
    {
      "supercategory": "vehicle",
      "id": 2,
      "name": "bicycle"
    },...]
"""
class COCO_Data(Dataset):
    def __init__(self, image_path, json_path):
        self.image_path = image_path
        self.json_path = json_path
        self.json_data = json.load(open(json_path, 'r'))
        self.image_data = self.json_data['images']
        self.annotation_data = self.json_data['annotations']
        self.category_data = self.json_data['categories']
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.image_data}
        self.cato_id_to_name = {cat['id']: cat['name'] for cat in self.category_data}
        self.cato_name_to_id = {cat['name']: cat['id'] for cat in self.category_data}
        self.img_id_to_annos = {}
        for ann in self.annotation_data:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_annos:
                self.img_id_to_annos[img_id] = []
            self.img_id_to_annos[img_id].append(ann)

    def check_image_path(self, index):
        img_info = self.image_data[index]
        img_id = img_info['id']
        img_name = self.img_id_to_filename[img_id]
        img_path = os.path.join(self.image_path, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found")
            return False
        return True
    
    def get_file_id2name(self):
        return self.img_id_to_filename
    
    def get_category_id2name(self):
        return self.cato_id_to_name
    
    def get_category_name2id(self):
        return self.cato_name_to_id

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx):
        img_info = self.image_data[idx]
        img_id = img_info['id']
        img_name = self.img_id_to_filename[img_id]
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path).convert('RGB')
        annos = self.img_id_to_annos.get(img_id, [])
        target = []
        for ann in annos:
            cat_id = ann['category_id']
            cat_name = self.cato_id_to_name[cat_id]
            bbox = ann['bbox']
            area = ann['area']
            target.append({'id': ann['id'], 'file_path': img_path, 'category_id': cat_id, 'category_name': cat_name, 'bbox': bbox, 'area': area, "iscrowd": ann.get("iscrowd", 0)})
        return {'image': img, 'target': target, 'id': img_id, 'img_info': img_info}
    
    def __getitems__(self, list_idx):
        results = []
        for idx in list_idx:
            temp_dict = self.__getitem__(idx)
            results.append(temp_dict)
        return results
  

def infer_model(model, tokenizer, image_processor, question, image_path, args, region_masks=None):
    """进行模型推理，输入问题和图片，返回模型输出"""
    # 读取并处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt', do_resize=True, 
                                              do_center_crop=False, size=[args.image_h, args.image_w])['pixel_values'][0]
    prompt = None
    # 构建输入ID
    if model.config.mm_use_im_start_end:
        prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{question}"
    else:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    # print(f"prompt: {prompt}")

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # 设置停止条件
    stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # 将区域掩码转换为CUDA tensor
    if region_masks is not None:
        region_masks = [[region_mask_i.cuda().half() for region_mask_i in region_masks]]
    else:
        region_masks = None

    # 模型推理
    with torch.inference_mode():
        model.orig_forward = model.forward
        model.forward = partial(model.orig_forward, region_masks=region_masks)
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
        model.forward = model.orig_forward

    # 解码输出
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # print(f"raw output: {outputs}")
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
        
    return outputs.strip()

import re



def find_target_bboxes(input_str, target_str=None):
    """
    在字符串中找到目标子字符串，并返回其后连续的bbox列表。
    
    参数:
        input_str (str): 输入字符串，可能包含bbox。
        target_str (str): 目标子字符串。
        
    返回:
        list: 连续bbox的列表，每个bbox为一个包含4个float数字的列表。
    """
    # 正则表达式匹配bbox格式：[float, float, float, float]
    bbox_pattern = r'\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]'
    input_str = input_str.lower()
    if target_str is not None:
        target_str = target_str.lower()

        # 找到目标子字符串的位置
        target_index = input_str.find(target_str)
        if target_index == -1:
            return []  # 若未找到目标字符串，返回空列表
        
        # 从目标字符串开始切片
        sliced_str = input_str[target_index + len(target_str):]

    else:
        sliced_str = input_str
    # 在切片后寻找所有bbox
    matches = re.findall(bbox_pattern, sliced_str)
    
    # 判断连续性，确保bbox是相邻的
    result = set()
    current_index = 0  # 初始从切片的开头
    
    for match in matches:
        # 格式化当前bbox为字符串形式
        bbox_str = f"[{', '.join(match)}]"
        next_index = sliced_str.find(bbox_str, current_index)
        
        # 如果当前bbox和上一个bbox是连续的（中间没有其他字符）
        if next_index == current_index or current_index == 0:
            # print(str(list(map(float, match))))
            result.add(str(list(map(float, match))))
            current_index = next_index + len(bbox_str)
        else:
            for i in range(current_index, next_index):
                if sliced_str[i]!= " ":
                    break
                if i == next_index - 1:  # 若所有字符都为空格，则认为是连续的
                    # print(str(list(map(float, match))))
                    result.add(str(list(map(float, match))))
                    current_index = next_index + len(bbox_str)
    result_lst = [eval(ele) for ele in result]


    return result_lst

# 示例用法
# input_str = "The person [559, 350, 691, 737] [389, 467, 469, 659] in the image is performing a skateboard trick [389, 467, 469, 659] [500, 657, 581, 757] in a parking lot [6, 549, 989, 992]. Person is [559, 350, 691, 737] [389, 467, 469, 659]"
# target_str = "person"
# print(find_target_bboxes(input_str, target_str))
"""
image_root_path = "/comp_robot/liushilong/data/coco/"
    anno_path = "/comp_robot/liushilong/data/coco/annotations/lvis_v1_minival_inserted_image_name_modify_catname.json"
"""
prompt_template_0 = """What is the location of {} in the image?"""
prompt_template_1 = """What is the location of all instances of categories, in the image? Please answer me respectively.
Category: {}"""
prompt_templates = [prompt_template_0, prompt_template_1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/comp_robot/yangyuqin/workplace/Multi-model/models/ml-ferret/checkpoints/target/ferret-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default="/comp_robot/liushilong/data/coco/val2017")
    parser.add_argument("--json_path", type=str, default="/comp_robot/liushilong/data/coco/annotations/instances_val2017.json")
    # parser.add_argument("--answers-file", type=str, default="lvis_result/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="ferret_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=336)  #  224
    parser.add_argument("--image_h", type=int, default=336)  #  224
    parser.add_argument("--add_region_feature", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--region_format", type=str, default="box", choices=["point", "box", "segment"])
    parser.add_argument("--result_path", type=str, default="/comp_robot/yangyuqin/workplace/Multi-model/result/lvis_obj_detection/ferret-7b.json")
    parser.add_argument("--is_query_all", type=int, default=1, help="Whether to Query [all categories](1) or [each category](0) for each image")

    args = parser.parse_args()
    print("Start")
    prompt_template = prompt_templates[args.is_query_all]

    # Data
    print("Loading data")
    dataset = COCO_Data(json_path=args.json_path, image_path=args.image_path)
    category_id2name = dataset.get_category_id2name()
    category_name2id = dataset.get_category_name2id()
    file_id2name = dataset.get_file_id2name()
    
    # Model

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loading model {model_name}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()
    # question_template = "Where is the object of {}?"
    # question_template = "What is the location of {} in the image?"
    print("Start inference")
    results = []
    empty_img_ids = []

    for element_idx, element in tqdm(enumerate(dataset)):
        # if element_idx >= 100:
        #     break
        target = element['target']
        img_info = element['img_info']
        img_id = img_info['id']
        img_basename = img_info['file_name']
        image_path = os.path.join(args.image_path, img_basename)
        # if not isinstance(element, dict):
        #     print(f"No object in the picture")
        #     print(f"img_basename: {img_basename}")
        #     # print(f"image_path: {image_path}")
        #     empty_img_ids.append(element)
        #     continue
        
        if len(target)==0:
            result_element = dict(id=img_id, file_name=img_info['file_name'], file_path=image_path, annotations=[], response="")
            results.append(result_element)
            print(f"image_path: {image_path}")
            print("No object in the picture")
            empty_img_ids.append(img_id)
            continue

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found")
            continue


        category_ids = [target_element['category_id'] for target_element in target]
        category_names = list(set(category_id2name[cato_id] for cato_id in category_ids))
        category_names = [category_name for category_name in category_names if category_name in category_name2id]


        # Inference for each category
        rect_set = set()
        new_annotations = []
        outputs_lst = []
        result_element = None

        if args.is_query_all == 1:
            # Inference for all categories
            ann_lst = []
            text = None
            question = prompt_template.format(str(category_names))
            outputs = infer_model(model, tokenizer, image_processor, question, image_path, args, region_masks=None)
            outputs_lst.append(outputs)
            try:
                annos = json.loads(outputs)
                if not isinstance(annos, list):
                    print(f"image_path: {image_path}")
                    print(f"Type Error in parsing annotations: {annos}")
                    continue
                advised_annos = []
                for ann in annos:
                    annotation = {}
                    x1, y1, x2, y2 = ann['rect']
                    annotation['rect'] = [int(x1*1000), int(y1*1000), int((x2-x1)*1000), int((y2-y1)*1000)]
                    annotation['class_id'] = category_name2id[ann['class']]
                    annotation['class'] = ann['class']
                    advised_annos.append(annotation)
                ann_lst = ann_lst + advised_annos
            except:
                print(f"image_path: {image_path}")
                print(f"Error in parsing annotations: {outputs}")
        elif args.is_query_all == 0:
            for category in category_names:
                question = prompt_template.format(category)
                outputs = infer_model(model, tokenizer, image_processor, question, image_path, args, region_masks=None)
                outputs_lst.append(outputs)

                anno_lst = find_target_bboxes(outputs, category)
                for i, anno in enumerate(anno_lst):
                    new_ann = {}
                    new_ann['class_id'] = category_name2id[category]
                    new_ann['class'] = category
                    x1, y1, x2, y2 = anno
                    new_ann['rect'] = [x1, y1, x2-x1, y2-y1]
                    if str(new_ann['rect']) in rect_set:
                        continue
                    rect_set.add(str(new_ann['rect']))
                    new_annotations.append(new_ann)
        result_element = dict(id=img_id, file_name=os.path.basename(image_path), file_path=image_path, annotations=new_annotations, response=outputs_lst)
        results.append(result_element)


    # Write results to file
    if args.is_query_all == 1:
        result_file_basename = os.path.basename(args.result_path).replace(".json", "_all.json")
        args.result_path = os.path.join(os.path.dirname(args.result_path), result_file_basename)
    print(f"Write results to file {args.result_path}")
    with open(args.result_path, 'w') as f:
        json.dump(results, f, indent=4)

    # White empty images
    empty_file_basename = os.path.basename(args.result_path).replace(".json", "_empty.json")
    empty_file_path = os.path.join(os.path.dirname(args.result_path),"empty", empty_file_basename)
    print(f"Write empty images to file {empty_file_path}")
    with open(empty_file_path, 'w') as f:
        json.dump(empty_img_ids, f, indent=4)

    print(f"Inference finished, total {len(results)} images in the {len(dataset)} elements of the dataset")
