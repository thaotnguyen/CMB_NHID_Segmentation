
import numpy as np
import cv2

def pad_volume(vol, newshape = (256, 256, 256), slice_axis = 2):
    '''
    This function pads the input volume according to the provided newshape. This function will not pad the axis
    along which slicing will take place.
    For example, if the slicing axis is 2, then this function will pad only axis 0 and 1.

    :param vol: volume whose shape is to be checked/padded
    :param newshape: (int tuple) The new shape of the volume
    :return: a volume whose shape conforms to a specified size
    '''

    volshape = vol.shape

    pad_size = [-1, -1, -1]

    # Pad the volume using numpy.pad() with padding of whatever the edge value is
    if volshape[0] < newshape[0] and slice_axis != 0:
        padding = (int)(np.floor((newshape[0] - volshape[0]) / 2))
        vol = np.pad(vol, ((padding, padding), (0, 0), (0, 0)), mode = "constant", constant_values=((0, 0), (0, 0), (0, 0))) #mode = "edge")
        pad_size[0] = padding

    if volshape[1] < newshape[1] and slice_axis != 1:
        padding = (int)(np.floor((newshape[1] - volshape[1]) / 2))
        vol = np.pad(vol, ((0, 0), (padding, padding), (0, 0)), mode = "constant", constant_values=((0, 0), (0, 0), (0, 0))) #mode = "edge")
        pad_size[1] = padding

    if volshape[2] < newshape[2] and slice_axis != 2:
        padding = (int)(np.floor((newshape[2] - volshape[2]) / 2))
        vol = np.pad(vol, ((0, 0), (0, 0), (padding, padding)), mode = "constant", constant_values=((0, 0), (0, 0), (0, 0))) #mode = "edge")
        pad_size[2] = padding

    return vol, pad_size




def return_as_list_of_strings(string_data):
    # First remove the square brackets
    s = string_data.replace("[", "").replace("]", "")

    # Then tokenize/split based on the comma
    s = s.split(",")
    if s[0] == "" and len(s) == 1:
        return []

    # Then create an empty list and the tokens to the list
    ret = list()
    for i in range(len(s)):
        ret.append(s[i])

    return ret



def return_as_list_of_ints(string_data):
    # First remove the square brackets
    s = string_data.replace("[", "").replace("]", "")

    # Then tokenize/split based on the comma
    s = s.split(",")
    if s[0] == "" and len(s) == 1:
        return []

    # Then create an empty list and the tokens to the list
    ret = list()
    for i in range(len(s)):
        ret.append(int(s[i]))

    return ret




def return_as_boolean(string_data):
    if (string_data.lower() == "true"):
        return True
    else:
        return False
    
def clipped_zoom(img, zoom_factor=0):

    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor == 0:
        return img


    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    
    result = cv2.resize(cropped_img, (resize_width, resize_height))
    
    if len(result.shape) == 2:
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] * (img.ndim - 2)
    else:
        pad_spec = [((pad_height1, pad_height2)), (pad_width1, pad_width2), (0,0)] * (img.ndim - 2)
    
    result = np.pad(result, pad_spec, mode='minimum')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def visual_check_datagen_reader(reader_dataset, path):
    from matplotlib import pyplot as plt

    for item in range(3):
        x, y = reader_dataset.__getitem__(item)

        for i in range(x.shape[0]):
            plt.figure(figsize = (16, 8))

            # plt.subplot(1, 2, 1)
            # plt.imshow(x[i, :, :, 0], cmap = "gray")
            # plt.imshow(np.ma.masked_where(y[i, :, :, 0] == 0, y[i, :, :, 0]), cmap="gist_rainbow", interpolation="nearest")
            # plt.title("T2")

            plt.subplot(1, 2, 2)
            plt.imshow(x[i, :, :, 0], cmap = "gray")
            plt.imshow(np.ma.masked_where(y[i, :, :, 0] == 0, y[i, :, :, 0]), cmap="gist_rainbow", interpolation="nearest")
            plt.title("T1")

            plt.suptitle("Unique values for class label 0: {}". format(np.unique(y[i, :, :, 0])))

            print(f'Writing images/{item}_{i}.png\r', end='')
            plt.savefig(f'/home/ttn/Development/CMB_NHID_Segmentation/image_proofs/{path}/{item}_{i}.png')
            plt.close()

