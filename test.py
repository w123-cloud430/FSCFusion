import cv2
import numpy as np
import os
import torch
import time
from PIL import Image, ImageOps
from models.vmamba_Fusion_efficross import VSSM_Fusion as net
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = net(in_chans=1)  # 注意参数名是in_chans，不是in_channel
model_path = "./model_last/my_cross/fusion_model.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.to(device)
    # 创建虚拟输入以初始化freq_align模块
    dummy_ir = torch.zeros(1, 1, 256, 256).to(device)
    dummy_vis = torch.zeros(1, 1, 256, 256).to(device)
    with torch.no_grad():
        dummy_ir_embed = model.patch_embed1(dummy_ir)
        dummy_vis_embed = model.patch_embed2(dummy_vis)
        _ = model.freq_align(dummy_ir_embed, dummy_vis_embed)
    # 加载权重
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
else:
    state_dict = torch.load(model_path, map_location='cpu')
    # 创建虚拟输入以初始化freq_align模块
    dummy_ir = torch.zeros(1, 1, 256, 256)
    dummy_vis = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        dummy_ir_embed = model.patch_embed1(dummy_ir)
        dummy_vis_embed = model.patch_embed2(dummy_vis)
        _ = model.freq_align(dummy_ir_embed, dummy_vis_embed)
    model.load_state_dict(state_dict)
# if use_gpu:
#     model = model.to(device)
#     state_dict = torch.load(model_path)
#     # 过滤掉不匹配的键
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         if 'freq_align' not in k:  # 跳过 freq_align 相关的键
#             new_state_dict[k] = v
#     model.load_state_dict(new_state_dict, strict=False)  # strict=False 允许部分加载
# else:
#     state_dict = torch.load(model_path, map_location='cpu')
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         if 'freq_align' not in k:
#             new_state_dict[k] = v
#     model.load_state_dict(new_state_dict, strict=False)

def imresize(arr, size, interp='bilinear', mode=None):
    numpydata = np.asarray(arr)
    im = Image.fromarray(numpydata, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)


def resize(image1, image2, crop_size_img, crop_size_label):
    image1 = imresize(image1, crop_size_img, interp='bicubic')
    image2 = imresize(image2, crop_size_label, interp='bicubic')
    return image1, image2


def get_image_files(input_folder):
    valid_extensions = (".bmp", ".tif", ".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])


def fusion(input_folder_ir, input_folder_vis, output_folder, resized_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建调整后图片的保存文件夹
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
        # 在调整后图片文件夹下创建子文件夹分别保存红外和可见光调整后的图片
        os.makedirs(os.path.join(resized_folder, 'ir'))
        os.makedirs(os.path.join(resized_folder, 'vis'))

    tic = time.time()
    # criteria_fusion = Fusionloss()

    ir_images = get_image_files(input_folder_ir)
    vis_images = get_image_files(input_folder_vis)

    for ir_image, vis_image in zip(ir_images, vis_images):
        path1 = os.path.join(input_folder_ir, ir_image)
        path2 = os.path.join(input_folder_vis, vis_image)

        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)  # 红外图
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)  # 可见光图

        # 调整图片尺寸
        target_size = [256, 256]  # 可以修改为你需要的尺寸
        img1_resized, img2_resized = resize(img1, img2, target_size, target_size)

        # 保存调整后的图片
        cv2.imwrite(os.path.join(resized_folder, 'ir', ir_image), img1_resized)
        cv2.imwrite(os.path.join(resized_folder, 'vis', vis_image), img2_resized)

        # 后续处理步骤
        img1_norm = np.asarray(img1_resized, dtype=np.float32) / 255.0
        img2_norm = np.asarray(img2_resized, dtype=np.float32) / 255.0

        img1_norm = np.expand_dims(img1_norm, axis=0)
        img2_norm = np.expand_dims(img2_norm, axis=0)

        img1_tensor = torch.from_numpy(img1_norm).unsqueeze(0).to(device)
        img2_tensor = torch.from_numpy(img2_norm).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(img1_tensor, img2_tensor)
            # 处理输出
            ones = torch.ones_like(out)
            zeros = torch.zeros_like(out)
            out = torch.where(out > ones, ones, out)
            out = torch.where(out < zeros, zeros, out)

            out_np = out.cpu().numpy()
            out_np = (out_np - np.min(out_np)) / (np.max(out_np) - np.min(out_np))

        d = np.squeeze(out_np)
        result = (d * 255).astype(np.uint8)

        output_filename = os.path.splitext(ir_image)[0] + os.path.splitext(ir_image)[1]
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result)

    toc = time.time()
    print('Processing time: {}'.format(toc - tic))


if __name__ == '__main__':
    input_folder_1 = '/root/autodl-tmp/FusionMamba_last/images/ir_1'  # 红外图输入文件夹
    input_folder_2 = '/root/autodl-tmp/FusionMamba_last/images/vis_1'  # 可见光图输入文件夹
    output_folder = './outputs'  # 融合结果输出文件夹
    resized_folder = './resized_images'  # 调整后图片保存文件夹

    fusion(input_folder_2, input_folder_1, output_folder, resized_folder)
    print("ok")
