import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import cv2
from PIL import Image
import pandas as pd

from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import matplotlib
import streamlit as st
from torchvision import transforms

def predict_hse(im_path):
    # load model
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: False
    model_cone = torch.hub.load(
    "ultralytics/yolov5", "custom", path = 'cone.pt', force_reload=True, autoshape=True
    ) # force_reload = recache latest code
    model_cone.conf = 0.7 # confidence threshold (0-1)

    # load model
    model_vest = torch.hub.load(
    "ultralytics/yolov5", "custom", path = 'vest.pt', force_reload=True, autoshape=True
    ) # force_reload = recache latest code
    model_vest.conf = 0.2 # confidence threshold (0-1)

    # load model
    model_helmet = torch.hub.load(
    "ultralytics/yolov5", "custom", path = 'hat.pt', force_reload=True, autoshape=True
    ) # force_reload = recache latest code
    model_helmet.conf = 0.6 # confidence threshold (0-1)
    im1 = im_path  # PIL image
    # im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
    # Inference
    results_cone = model_cone([im1], size=640) # batch of images
    results_helmet = model_helmet([im1], size=640) # batch of images
    results_vest = model_vest([im1], size=640) # batch of images
    # Results
    df1 = results_cone.pandas().xyxy[0]  # im1 predictions (pandas)
    df2 = results_helmet.pandas().xyxy[0]  # im1 predictions (pandas)
    df3 = results_vest.pandas().xyxy[0]  # im1 predictions (pandas)
    result = pd.concat([df1, df2,df3], axis=0)
    st.write(result)
    item_counts = result["name"].value_counts()
    # generate color
    colors = [['cone', 'green'], ['helmet', 'purple'], ['with_vest', 'blue'],['head','yellow'],['no_vest', 'red']]
    colors = pd.DataFrame(colors, columns=['name', 'color'])
    result = pd.merge(result, colors, on='name')
    # read input image
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    img = transform(im1)
    bbox = result.iloc[:,:4].values.tolist()
    labels = result['name'].values.tolist()
    colors = result['color'].values.tolist()
    bbox = torch.tensor(bbox, dtype=torch.int)
    # draw bounding box on the input image
    img=draw_bounding_boxes(img, bbox, width=3, labels=labels, colors=colors,font_size=100, fill=True)
    # transform it to PIL image and display
    img = torchvision.transforms.ToPILImage()(img)
    return img, item_counts



img = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
if img is not None:
    file_details = {"Filename":img.name,"FileType":img.type,"FileSize":img.size}
    image = Image.open(img)
    st.image(image,use_column_width=True,caption='Input')
    resim, stats = predict_hse(image)
    st.image(resim, use_column_width=True,caption='Output')
    st.write(stats)

