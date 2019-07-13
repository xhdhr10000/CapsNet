# coding: utf-8
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image, ImageOps
from keras import utils

class Dataset():
    def __init__(self, dataset_path):
        self.size = 256
        self.names = {}
        self.boxes = self.load_bbox(dataset_path)
        self.count = len(self.boxes)
        self.num_classes = len(self.names)
        self.seq = np.arange(self.count)
        np.random.shuffle(self.seq)
        self.pseq = 0

    def load_bbox(self, dataset_path):
        dataset = []
        bbox_path = os.path.join(dataset_path, 'bbox')
        for d in os.listdir(bbox_path):
            path = os.path.join(bbox_path, d)
            if not os.path.isdir(path): continue
            for boxfile in os.listdir(path):
                if not boxfile.endswith('xml'): continue
                tree = ET.parse(os.path.join(path, boxfile))
                root = tree.getroot()
                filename = os.path.join(dataset_path, 'img', d, boxfile.split('.')[0]+'.JPEG')
                for o in root.findall('object'):
                    name = o.find('name').text
                    if not name in self.names.keys(): self.names[name] = len(self.names)
                    box = []
                    bndbox = o.find('bndbox')
                    box.append(int(bndbox.find('xmin').text))
                    box.append(int(bndbox.find('ymin').text))
                    box.append(int(bndbox.find('xmax').text))
                    box.append(int(bndbox.find('ymax').text))
                    dataset.append({'filename': filename, 'id': self.names[name], 'box': box})
        print('load_bbox finished. count=%d' % len(dataset))
        return dataset

    def resize(self, im):
        w, h = im.size
        dw = max(w,h) - w
        dh = max(w,h) - h
        padding = (dw//2, dh//2, dw-dw//2, dh-dh//2)
        im = ImageOps.expand(im, padding)
        im = im.resize((self.size, self.size))
        return im

    def load_image(self, batch_size):
        x = []
        y = []
        for i in range(batch_size):
            box = self.boxes[self.seq[self.pseq + i]]
            filename = box['filename']
            img = Image.open(filename)
            im = img.crop(box['box'])
            im = self.resize(im)
            im = np.asarray(im, dtype=np.float32) / 255.0
            x.append(im)
            y.append(box['id'])
            img.close()
        self.pseq += batch_size
        x = np.array(x)
        y = np.array(y)
        y = utils.to_categorical(y, self.num_classes)
        return x, y

if __name__ == '__main__':
    dataset = Dataset('../dataset')
    print('names: %d' % len(dataset.names))
    for i in range(10):
        images, labels = dataset.load_image(32)
        print(images.shape)
        print(labels.shape)