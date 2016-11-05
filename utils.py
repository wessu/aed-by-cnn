import json
import csv
import numpy as np
import os

# Upscale array
def shift(array_list, shift_size, axis):
    n_axes = len(array_list[0].shape)
    obj = [slice(None, None, None) for ii in range(n_axes)]
    obj[axis] = slice(shift_size, None, 1)
    obj = tuple(obj)

    pad_width = [(0, 0) for ii in range(n_axes)]
    pad_width[axis] = (0, shift_size)

    out_array_list = [np.pad(array[obj], pad_width, 'constant')
                      for array in array_list]
    return out_array_list


def upscale(func, input_list, method='naive', scale_factor=1,
            in_axis=2, out_axis=2):
    '''
    array: numpy.array

    method: str
        'naive' or 'patching'

    scale_factor: int

    '''

    assert(method in ['naive', 'patching'])
    if method == 'naive':
        array = func(*input_list)[-1]
        new_array = np.repeat(array, scale_factor, axis=out_axis)
    elif method == 'patching':
        output_list = [func(*shift(input_list, ii, axis=in_axis))[0]
                       for ii in range(scale_factor)]
        output = np.stack(output_list, axis=out_axis+1)

        new_shape = list(output_list[0].shape)
        new_shape[out_axis] = -1
        new_shape = tuple(new_shape)

        new_array = np.reshape(output, new_shape)

    return new_array

def get_annot_files(file_name, class_num):
    class_dict = ['air_conditioner', 
                  'car_horn', 
                  'children_playing', 
                  'dog_bark',
                  'drilling',
                  'engine_idling',
                  'gun_shot',
                  'jackhammer',
                  'siren',
                  'street_music']
    root = os.path.join('/home/dovob/NAS/Database/UrbanSound/data', class_dict[class_num])
    return os.path.join(root, file_name+'.csv'), os.path.join(root, file_name+'.json')

def load_us_annotation(file_name, class_num, length, hop_size=512, sr=44100):
    def time_2_frame(time):
        return int(time * sr // hop_size)

    csv_fp, json_fp = get_annot_files(file_name, class_num)
    if not os.path.exists(csv_fp):
        return np.zeros((10, length))
        
    event_list = []
    with open(csv_fp) as csvfile:
        r = csv.reader(csvfile)
        event_list = [(float(row[0]), float(row[1])) for row in r]
    # duration = 0
    # with open(json_fp) as jsonfile:
    #     data = json.load(jsonfile)
    #     duration = float(data['duration'])

    # assert(duration > 0)
    # duration = time_2_frame(duration)
    duration = length
    annotation = np.zeros((10, duration))

    for on_beat, down_beat in event_list:
        on_beat = time_2_frame(on_beat)
        down_beat = time_2_frame(down_beat)
        if down_beat >= duration:
            down_beat = duration - 1
        if on_beat > down_beat:
            on_beat = down_beat
        for i in range(on_beat, down_beat): 
            annotation[class_num][i] = 1.0 
            
    return annotation

def load_us8k_annotation(file_name, class_num, length):
    annotation = np.zeros((10, length))
    for i in range(length):
        annotation[class_num][i] = 1.0
            
    return annotation

##### Iterate input list with size #####
def my_iterator(inputs, bsize):
    k = 1 if len(inputs) % bsize > 0 else 0
    for idx in range(0, len(inputs)/bsize + k):
        sl = slice(idx*bsize, (idx+1)*bsize)
        yield (inputs[sl], idx)

