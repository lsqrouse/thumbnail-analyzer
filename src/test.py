import tensorflow as tf

#
# Original code based on two articles from Jonathan Leban:
# Part 1: https://towardsdatascience.com/image-recognition-with-machine-learning-on-python-image-processing-3abe6b158e9a
# Part 2: https://towardsdatascience.com/image-recognition-with-machine-learning-on-python-convolutional-neural-network-363073020588
#
def get_dataset(image_paths, image_type, resize_shape, channels):
    filename_tensor = tf.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)
    
    def _map_fn(filename):
        decode_images = decode_image(filename, image_type, resize_shape, channels=channels)
        return decode_images
    
    map_dataset = dataset.map(_map_fn) # we use the map method: allow to apply the function _map_fn to all the 
    # elements of dataset 
    return map_dataset

def decode_image(filename, image_type, resize_shape, channels):
    value = tf.io.read_file(filename)
    if image_type == 'png':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    
    if resize_shape is not None and image_type in ['png', 'jpeg']:
        decoded_image = tf.image.resize(decoded_image, resize_shape)
    
    return decoded_image


def get_image_data(image_paths, image_type, resize_shape, channels):
    dataset = get_dataset(image_paths, image_type, resize_shape, channels)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_image = iterator.get_next()
    
    return next_image

data = get_image_data(['./img/soccer_ball.jpg'], 'jpg', 0, 0)

print(data)