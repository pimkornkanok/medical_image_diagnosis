"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
import numpy as np
import torch
import streamlit as st
import pandas as pd
import os
import json
import pretrainedmodels
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from torch import nn
from tqdm import tqdm
import torchfunc
import plotly.express as px 
import matplotlib.pyplot as plt
from pathlib import Path



#############################################################################################################
# helpful to reset memory usage and avoid cuda segfaults
torchfunc.cuda.reset()

header = st.container()
model = st.container()
col1, col2 = st.columns(2)

# Get the absolute path of the root directory
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Button to open the model page

model_file_path = os.path.join(root_path, "best_model_efficient.pth")

def load_model(model_path, model_name, num_classes):
    """loads a torch classification model
    Args:
        - model_path(str): path to the model .pth
        - model_name(str): name of the model type, such as efficientnet-b2
        - num_classes(int): number of output classes
    Returns:
        - model(torch model): loaded classification model
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if "efficientnet" in model_name:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    else:
        model = get_model(model_name, num_classes)

def pre_transforms(image_size=380):
    """
    Converts the image to a square of size image_size x image_size
    (keeping aspect ratio)

    Args:
        - image_size(int): both the height and width to which to resize the
        input image

    Returns:
      - Augmented image

    """
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=0),
    ]
    return result

def post_transforms():
    """
    The transforms perform ImageNet image normalization and convert the image
    to a torch.Tensor.

    Returns:
        List[albu.BasicTransform]: A list of post-processing transforms for image data.
    """
    return [albu.Normalize(), ToTensorV2()]


def compose(transforms_to_compose):
    """
    combines all input augmentations into one single pipeline

    Returns:
        A list of post-processing transforms for image data.
    """
    result = albu.Compose(
        [item for sublist in transforms_to_compose for item in sublist]
    )
    return result


def get_model(model_name: str, num_classes: int, pretrained: str = "imagenet"):
    """
    Loads a pretrained model and switches its last layer to suit the number
    of classes of the current dataset

    Args:
        - model_name (str): the name of the model to import from
        pretrainedmodels
        - num_classes (str): the number of classes in your dataset
        - pretrained (str): which pretrained weights to use
    Returns:
        - model
    """
    model_fn = pretrainedmodels.__dict__[model_name]
    model = model_fn(num_classes=1000, pretrained=pretrained)
    model.fc = nn.Sequential()
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    return model


def load_model(model_path, model_name, num_classes):
    """loads a torch classification model
    Args:
        - model_path(str): path to the model .pth
        - model_name(str): name of the model type, such as efficientnet-b2
        - num_classes(int): number of output classes
    Returns:
        - model(torch model): loaded classification model
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if "efficientnet" in model_name:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    else:
        model = get_model(model_name, num_classes)

    # try:
    #     state_dict = torch.load(model_path, map_location=device)["model_state_dict"]
    #     state_dict["_fc.weight"] = state_dict.pop("_fc.1.weight")
    #     state_dict["_fc.bias"] = state_dict.pop("_fc.1.bias")
    #     model.load_state_dict(state_dict)
    # except FileNotFoundError:
    #     print("Model file not found")
    #     exit(1)
    # except KeyError:
    #     print("Model state_dict is not valid")
    model.to(device)
    model.eval()
    return model



def main_loop(original_image, original_file_name):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    transform = compose(
        [

            pre_transforms(image_size=350),
            post_transforms(),
        ]
    )

    tag_to_label = json.load(open('labels.json'))
    #print("label tags", tag_to_label)
    labels_to_names = {v: k for k, v in tag_to_label.items()}
    # Number of classes
    num_classes = len(tag_to_label.keys())

    model = load_model(model_file_path, 'efficientnet-b2', num_classes)


    with torch.no_grad():
        transformed = transform(image=original_image)
        transformed_image = transformed["image"]
        input_network = transformed_image.to(device)
        output_network = model(input_network[None, ...])
        probabilities = (
            torch.softmax(torch.from_numpy(output_network.cpu().numpy()[0]), dim=0)
            .cpu()
            .numpy()
        )
        label = probabilities.argmax().item()
        displayed_label = labels_to_names[label]
        #print('Predicted label: ', displayed_label)
        # Create a list of class names
        # Create a DataFrame with class names and probabilities
        data = {
            'Class': [labels_to_names[i] for i in range(len(labels_to_names))],
            'Probability': probabilities,
            'image_name': original_file_name
        }
        df = pd.DataFrame(data)

        # Set the 'Class' column as the index
        df = df.set_index('Class')
        #print(df)
        # Create the horizontal bar plot
        # Plotly
        st.success("The output is: ")
        fig=px.bar(df,x='Probability', orientation='h', color='Probability')
        fig.update_layout(title="<b>COVID_19 detection</b>")
        # Update the layout with bold font for the axis labels
        fig.update_layout(
            xaxis=dict(title="<b>Probability</b>"),
            yaxis=dict(title="<b>Class</b>")
        )
        fig.update_yaxes(tickfont_family="Arial Black", tickfont=dict(size=16))
        fig.update_xaxes(tickfont_family="Arial Black", tickfont=dict(size=16))

        st.write(fig)

   

if __name__ == '__main__':
    with header:
        st.title("COVID19 Detection from Chest X-Ray")
        #st.subheader("COVID19 detection from chest X-ray!")
        st.markdown("""The utilization of chest X-ray images for the detection of COVID-19 has become an important tool in the battle against the pandemic. By examining the distinctive patterns and abnormalities present in the X-ray scans of the chest, medical professionals can identify potential indications of the virus. This approach offers a non-invasive and relatively quick method for assessing the presence of COVID-19, providing valuable insights that can aid in early diagnosis and treatment. Leveraging advancements in image analysis and machine learning, researchers and healthcare practitioners continue to refine and improve this technique, 
        enhancing its accuracy and effectiveness in identifying COVID-19 cases from chest X-ray images.""")

    with model:
        image_file = st.file_uploader("Upload Chest X-ray Image for COVID19 Detection", type=['jpg', 'png', 'jpeg'])
        if not image_file:
            st.warning("No image file uploaded.")

        else:
            # Get the original file name of the uploaded image
            original_file_name = image_file.name

            original_image = Image.open(image_file)
            original_image = np.array(original_image)

            main_loop(original_image, original_file_name)




