import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np



def predict(image_path):
    # Load the image and resize it to match the model's expected input shape
    img = image.load_img(image_path, target_size=(62, 47))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for the batch

    # Preprocess the image for the model
    img_array = preprocess_input(img_array)

    # Load the trained model
    loaded_model = tf.keras.models.load_model('SQLite/trained_model.h5')

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # Get the result
    result = predictions.flatten()[0]

    print(result)
    return result

# Example usage:
image_path = '/Users/shahd.metwally/monorepo/SQLite/Arturo_Gatti_0002.jpg'
prediction_result = predict(image_path)
