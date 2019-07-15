# coding: utf-8
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image, ImageOps
from keras import utils
import imgaug.augmenters as iaa

class Dataset(utils.Sequence):
    def __init__(self, dataset_path, batch_size):
        self.size = 128
        self.batch_size = batch_size
        self.names = {}
        self.boxes = self.load_bbox(dataset_path)
        self.count = len(self.boxes)
        self.num_classes = len(self.names)
        self.seq = np.arange(self.count)
        np.random.shuffle(self.seq)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 2.0)), # blur images with a sigma of 0 to 2.0
            iaa.GammaContrast(gamma=(0.3, 2.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
        ])


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
        return self.count_train // self.batch_size

    def getImage(self, filename, box):
        img = Image.open(filename)
        im = img.crop(box)
        im = self.resize(im)
        im = np.asarray(im, dtype='uint8')
        img.close()
        return im

    def __getitem__(self, index):
        x = []
        y = []
        for i in range(self.batch_size):
            box = self.boxes[self.seq[(index * self.batch_size + i) % self.count_train]]
            im = self.getImage(box['filename'], box['box'])
            x.append(im)
            y.append(box['id'])
        x = np.array(x)
        x = self.aug.augment_images(x)
        x = x.astype('float32') / 255.0
        y = np.array(y)
        y = utils.to_categorical(y, self.num_classes)
        return x, y

    class EvalDataset(utils.Sequence):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return self.dataset.count_eval // self.dataset.batch_size

        def __getitem__(self, index):
            x = []
            y = []
            for i in range(self.dataset.batch_size):
                box = self.dataset.boxes[self.dataset.seq[(index * self.dataset.batch_size + i) % self.dataset.count_eval + self.dataset.count_train]]
                im = self.dataset.getImage(box['filename'], box['box'])
                x.append(im)
                y.append(box['id'])
            x = np.array(x)
            x = x.astype('float32') / 255.0
            y = np.array(y)
            y = utils.to_categorical(y, self.dataset.num_classes)
            return x, y

if __name__ == '__main__':
    bs = 4
    dataset = Dataset('../dataset', bs)
    print('names: %d' % len(dataset.names))
    for i in range(2):
        images, labels = dataset[i]
        print(images.shape)
        print(labels.shape)
        images *= 255.0
        for j in range(bs):
            Image.fromarray(images[j].astype('uint8'), mode='RGB').show()
