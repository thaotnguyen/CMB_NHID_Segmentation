
import numpy as np


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

def visual_check_datagen_reader(reader_dataset, path):
    from matplotlib import pyplot as plt

    for item in range(3):
        x, y = reader_dataset.__getitem__(item)

        for i in range(x.shape[0]):
            plt.figure(figsize = (16, 8))

            plt.subplot(1, 2, 1)
            plt.imshow(x[i, :, :, 0], cmap = "gray")
            plt.imshow(np.ma.masked_where(y[i, :, :, 0] == 0, y[i, :, :, 0]), cmap="gist_rainbow", interpolation="nearest")
            plt.title("T2")

            plt.subplot(1, 2, 2)
            plt.imshow(x[i, :, :, 1], cmap = "gray")
            plt.imshow(np.ma.masked_where(y[i, :, :, 0] == 0, y[i, :, :, 0]), cmap="gist_rainbow", interpolation="nearest")
            plt.title("T1")

            plt.suptitle("Unique values for class label 0: {}". format(np.unique(y[i, :, :, 0])))

            print(f'Writing images/{item}_{i}.png\r', end='')
            plt.savefig(f'/abyssaldata/projects/ePVS/images/{path}/{item}_{i}.png')
            plt.close()

