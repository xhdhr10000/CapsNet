# coding: utf-8
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image, ImageOps
from keras import utils

class Dataset(utils.Sequence):
    def __init__(self, dataset_path, batch_size):
        self.size = 64
        self.batch_size = batch_size
        self.names = {}
        self.boxes = self.load_bbox(dataset_path)
        self.count = len(self.boxes)
        self.num_classes = len(self.names)
        self.seq = np.arange(self.count)
        np.random.shuffle(self.seq)

        # Train eval split
        self.count_train = int(self.count*0.9)
        self.count_eval = self.count - self.count_train
        self.eval_dataset = self.EvalDataset(self)

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

    def __len__(self):
        return self.count_train

    def imageAugmentation(self, im):
        if np.random.randint(2):
            im = np.fliplr(im)
        if np.random.randint(2):
            im = np.flipud(im)
        if np.random.randint(2):
            im = np.rot90(im)
        return im

    def getImage(self, filename, box):
        img = Image.open(filename)
        im = img.crop(box)
        im = self.resize(im)
        im = np.asarray(im, dtype=np.float32) / 255.0
        im = self.imageAugmentation(im)
        img.close()
        return im

    def __getitem__(self, index):
        x = []
        y = []
        for i in range(self.batch_size):
            box = self.boxes[self.seq[(index + i) % self.count_train]]
            im = self.getImage(box['filename'], box['box'])
            x.append(im)
            y.append(box['id'])
        x = np.array(x)
        y = np.array(y)
        y = utils.to_categorical(y, self.num_classes)
        return x, y

    class EvalDataset(utils.Sequence):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return self.dataset.count_eval

        def __getitem__(self, index):
            x = []
            y = []
            for i in range(self.dataset.batch_size):
                box = self.dataset.boxes[self.dataset.seq[(index + i) % self.dataset.count_eval + self.dataset.count_train]]
                im = self.dataset.getImage(box['filename'], box['box'])
                x.append(im)
                y.append(box['id'])
            x = np.array(x)
            y = np.array(y)
            y = utils.to_categorical(y, self.dataset.num_classes)
            return x, y

if __name__ == '__main__':
    dataset = Dataset('../dataset', 1)
    print('names: %d' % len(dataset.names))
    for i in range(10):
        images, labels = dataset[i]
        print(images.shape)
        print(labels.shape)
        images *= 255.0
        Image.fromarray(images[0].astype('uint8'), mode='RGB').show()
