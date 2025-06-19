import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load the CNN model (replace with your model path)
model = load_model("weed_classifier.h5")  # Path to your CNN .h5 model

# Define the image size your model expects (224x224 based on your model)
IMG_SIZE = (224, 224)

def predict_image(img):
    # Ensure the image is in the correct size for the model (224x224)
    img = img.resize(IMG_SIZE)
    
    # Convert the image to a numpy array
    img_array = keras_image.img_to_array(img)
    
    # Add batch dimension and normalize image
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize image (as done during training)

    # Predict using the CNN model
    predictions = model.predict(img_array)
    
    # Get the predicted class index (highest probability)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Define the class names as per your training dataset (same as before)
    class_labels = [
        'Aerva lanata', 'Amaranthus spinosus123', 'Calotropis plant123', 'Chinee apple',
        'Cleome viscosa', 'Cyprus rotundus', 'Datura', 'Lantana', 'Martynia Annua111', 
        'Negative', 'Parkinsonia', 'Parthenium', 'Phyllanthus nirur', 'Prickly acacia', 
        'Pycerus polystachyos', 'Rubber vine', 'Siam weed', 'Snake weed', 'Sorghum halepense', 
        'Taraxacum officinale', 'Trianthema portulacastrum', 'Tridax procumbens', 'Xanthium strumarium', 
        'abutilon hirtum123', 'acalypha indica123', 'cuscuta', 'digitaria sanguinalis', 'echinochloa colona', 
        'eichhornia crassipe', 'euphorbia prostrat', 'portulaca oleracea', 'serpyllifolia123'
    ]
    
    # Map the predicted class index to the class label
    predicted_label = class_labels[predicted_class]

    return predicted_label
