import streamlit as st # type: ignore
from PIL import Image, ImageOps # type: ignore
import numpy as np # type: ignore
import lime # type: ignore
from lime import lime_image # type: ignore
from skimage.segmentation import mark_boundaries  # type: ignore # Import mark_boundaries function
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import shutil, os
from os import listdir
from os.path import isfile, join

@st.cache_resource

# This function is used to plot the explainable AI Image
def ImageExplainer(_prediction, xai_image_title, xai_image, image_to_explain):
  # Create a LimeImageExplainer object
  explainer = lime_image.LimeImageExplainer()

  # Generate an explanation for the image
  explanation = explainer.explain_instance(
    image_to_explain,
    _prediction,
    top_labels=5,
    hide_color=0,
    num_samples=1000
  )
  
  ind = explanation.top_labels[0]

  # Map each explanation weight to the corresponding superpixel
  dict_heatmap = dict(explanation.local_exp[ind])
  heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

  # Get the original image and the mask for the top features
  temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
  )
  
  folder = "explainableAIImage/"

  # Plot the original image to explain
  plt.figure(figsize=(10, 10))
  plt.suptitle(xai_image_title)
  plt.subplot(1, 3, 1)
  plt.imshow(image_to_explain)
  plt.title('Original MR Image')

  # Plot the superpixels (for the most positively contributing regions)
  plt.subplot(1, 3, 2)
  plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
  plt.title('Most positively \ncontributing regions')

  # Plot the heatmap (each region's contribution)
  plt.subplot(1, 3, 3)
  plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
  plt.colorbar()
  plt.title("Each region's contribution")
  plt.tight_layout()
  plt.savefig(folder + xai_image)
  
  saved_plot = Image.open(folder + xai_image)
  st.image(saved_plot, width=750)
  
def save_image_to_folder(directory, file):
  folder = 'image'
  dir = os.listdir(folder)
    
  # Checking if the list is empty or not
  if len(dir) == 0:
    # Save file into a folder
    with open(os.path.join(directory,file.name),"wb") as f: 
      f.write(file.getbuffer())  
  else:
    for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

    with open(os.path.join(directory,file.name),"wb") as f: 
      f.write(file.getbuffer())  
     
st.write(""" # Clinical Significance Classification and Disc Bulge Detection of Lumbar Spine MRI """)
  
tab1, tab2 = st.tabs(["Clinical Significance Classification", "Disc Bulge Detection"])

with tab1:
  st.header("Clinical Significance Classification")
  st.markdown("This section provides you with the ability to upload an MRI of a lumbar spine and get how clinically significant it's diagnosis will be")
  
  file = st.file_uploader("Please upload an MRI scan", type=["jpg", "png"])
  
  if file is None:
    st.text("Please upload an image file")
  else:
    saved_image = Image.open(file)
    st.image(saved_image, width=550)
    save_image_to_folder("image", file)   
      
    # Load model
    cs_model = load_model("models/cs.h5")
    get_file_from_saved_folder = [f for f in listdir("image") if isfile(join("image", f))]
    img = image.load_img('image/' + get_file_from_saved_folder[0], target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = cs_model.predict(img_array)

    class_labels = ['No', 'Mild', 'Serious']
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    condition = class_labels[predicted_class]

    st.write(f"<div style='padding: 0px 0 40px; font-size:24px;'>{class_labels[predicted_class]} clinical significance with a confidence of {confidence:.2f}%</div>", unsafe_allow_html=True)

    image_to_explain = img_array[0]

    def predict_cs(image_array):
      image_array = image_array.reshape(-1, 150, 150, 3)
      return cs_model.predict(image_array)

    ImageExplainer(predict_cs, "MR Image regions' contributions to the model's clinical significance classification", "cs_plot.png", image_to_explain)

with tab2:
  st.header("Disc Bulge Detection")
  
  file_2 = st.file_uploader("Please upload MRI scan", type=["png", "jpg"])
  st.set_option('deprecation.showfileUploaderEncoding', False)  

  if file_2 is None:
    st.text("Please upload an image file")
  else:
    saved_image_2 = Image.open(file_2)
    st.image(saved_image_2, width=550)
  
    save_image_to_folder("image", file_2)   
    
    # Load model
    bulge_model = load_model("models/bulge_model.h5")
    get_file_from_saved_folder = [f for f in listdir("image") if isfile(join("image", f))]
    img = image.load_img('image/' + get_file_from_saved_folder[0], target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    predictions = bulge_model.predict(img_array)

    if predictions[0][0] > 0.5:
      confidence = (predictions[0][0]) * 100
      st.write(f"<div style='padding: 0px 0 40px; font-size:24px;'>Prediction: No serious disc bulge detected with a confidence of : {confidence:.2f}%</div>", unsafe_allow_html=True)
      # st.write(f"Prediction: No serious disc bulge detected with a confidence of : {confidence:.2f}%")
    else:
      confidence = (1-predictions[0][0]) * 100
      st.write(f"<div style='padding: 0px 0 40px; font-size:24px;'>Prediction: Serious disc bulge detected with a confidence of: {confidence:.2f}%</div>", unsafe_allow_html=True)
      # st.write(f"Prediction: Serious disc bulge detected with a confidence of: {confidence:.2f}%")

    image_to_explain = img_array[0]
          
    def predict_disc_bulge(image_array):
      image_array = image_array.reshape(-1, 150, 150, 3)
      return bulge_model.predict(image_array)
    
    ImageExplainer(predict_disc_bulge, "MR Image regions' contributions to the model's disc bulge detection", "disc_bulge_plot.png", image_to_explain)