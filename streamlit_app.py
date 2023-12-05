import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
from torchvision.models import mobilenet_v3_large

# Page Configuration
st.set_page_config(
    page_title="FER - Classification",
    page_icon="🫠",
    layout="centered",
)

# Title and Description
st.title('Facial Expressions - Image Classification')
st.markdown('Upload an image and let the model predict the emotion!')

# Load Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = 7
model_path = os.path.abspath('mobilenetv3_large_100_checkpoint_fold4.pt')

# Load the MobileNetV3 model architecture
try:
    model = mobilenet_v3_large(weights=None, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Upload Image Section
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display Uploaded Image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    class_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Prediction Section
    if st.button('Make Prediction', key="prediction_button"):
        try:
            # Check if model is defined
            if model is not None:
                # Transformation
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])

                # Preprocess the image
                input_image = transform(Image.open(uploaded_image).convert('RGB')).unsqueeze(0)

                # Prediction class
                with torch.no_grad():
                    output = model(input_image)

                # Get the probability scores
                probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()

                # Get the index of the maximum probability
                max_index = np.argmax(probabilities)

                # Display prediction results
                st.markdown("## Prediction Result")
                for i in range(len(class_name)):
                    # Set the color to red if it's the maximum value, otherwise use the default color
                    color = "red" if i == max_index else None
                    st.write(f"### <span style='color:{color}'>{class_name[i]}: {probabilities[i]*100:.2f}%</span>", unsafe_allow_html=True)

            else:
                st.error("Model not loaded. Please check the error above.")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Member Section
st.sidebar.markdown("## Team Members")
team_members = '''
- Watayut Pankong (64112790)
- Tassawas Buathong (64127087)
- Chayathon Cheechang (64102999)
'''

st.sidebar.markdown(team_members, unsafe_allow_html=True)
