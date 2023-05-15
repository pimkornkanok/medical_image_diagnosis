import os
import streamlit as st
import subprocess

header = st.container()
l1 = st.container()
l2 = st.container()

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "../covid_images")
os.makedirs(IMAGES_DIR, exist_ok=True)


with header:
    st.title("COVID-19 detection from chest x-ray using deep learning")
    st.markdown("### COVID-19 CHEST X-RAY DATABASE")
    st.markdown("A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal and other lung infection dataset is released in stages. In the first release we have released 219 COVID-19, 1341 normal and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection) and 1345 Viral Pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients") 
    st.markdown(""" **COVID-19 data:**
COVID data are collected from different publicly accessible dataset, online sources and published papers.
 -2473 CXR images are collected from padchest dataset[1]. -183 CXR images from a Germany medical school[2]. -559 CXR image from SIRM, Github, Kaggle & Tweeter[3,4,5,6] -400 CXR images from another Github source[7].

""")
    st.markdown(""" 
 **Normal images:**
10192 Normal data are collected from from three different dataset. -8851 RSNA [8] -1341 Kaggle [9]

**Lung opacity images:**
6012 Lung opacity CXR images are collected from Radiological Society of North America (RSNA) CXR dataset [8]

**Viral Pneumonia images:**
1345 Viral Pneumonia data are collected from the Chest X-Ray Images (pneumonia) database.""")
    st.markdown("Please refer to Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238 for all references")
with col1:
	st.image(IMAGES_DIR + '/COVID19(467).jpg', width=300, caption='COVID-19')
with col2:
	st.image(IMAGES_DIR + '/Normal-4.png', width=300, caption='Normal')

with l1:
	st.markdown(""" 
**Distribution of images in the dataset** 

Total number of images in the dataset : 13808 : {'COVID': 3616, 'Normal': 10192}
Train data: 10808 : {'COVID': 2843, 'Normal': 7965}
Test data: 3000 : {'COVID': 773, 'Normal': 2227}""")  

with l2:
	st.markdown(" **Trained model  is efficient-net B2.  Test Accuracy: 99.1%**")
	
with col1:
	st.image(IMAGES_DIR + '/covid_data_1.png', width=300)
with col2:
	st.image(IMAGES_DIR + '/covid_data_2.png', width=400)
with col3:
	st.image(IMAGES_DIR + '/covid_data_3.png', width=400)
with col4:
	st.image(IMAGES_DIR + '/covid_data_4.png', width=400)
with col5:
	st.image(IMAGES_DIR + '/covid_data_5.png', width=400)




#st.title("Try Our AI Model")


# Get the absolute path of the root directory
#root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Button to open the model page
#if st.button("Click Here To Try Our AI Model"):
#    model_file_path = os.path.join(root_path, "Detect_COVID19.py")
#    subprocess.Popen(["streamlit", "run", model_file_path])


