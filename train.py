"""
Retrain the YOLO model for your own dataset.
"""
import sys
import os
import threading
import argparse
import time
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from multiprocessing.dummy import Pool as ThreadPool
import itertools

parser = argparse.ArgumentParser(description='')
parser.add_argument('--annotation_path', dest='annotation_path', default='train.txt', help="""annotation data.""")
parser.add_argument('--log_dir', dest='log_dir', default='logs/000/', help="""training log folder.""")
parser.add_argument('--classes_path', dest='classes_path', default='model_data/voc_classes.txt', help="""classes path.""")
parser.add_argument('--anchors_path', dest='anchors_path', default='model_data/yolo_anchors.txt', help="""anchors path.""")
parser.add_argument('--batch_size', dest='batch_size', default=64, help="""anchors path.""")
args = parser.parse_args()

try:
    threadpool = ThreadPool(args.batch_size)
except Exception as e:
    print(e)
    exit(1)

def _main():
    annotation_path = args.annotation_path
    log_dir = args.log_dir
    classes_path = args.classes_path
    anchors_path = args.anchors_path
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    checkpoint_refresh = ModelCheckpoint(os.path.join(log_dir, 'refresh_checkpoint.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    batch_size = args.batch_size

    train_data_factory = data_factory(lines[:num_train], batch_size, input_shape, anchors, num_classes)
    train_data_factory.start()

    valid_data_factory = data_factory(lines[num_train:], batch_size, input_shape, anchors, num_classes)
    valid_data_factory.start()

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper_new(train_data_factory),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper_new(valid_data_factory),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint, checkpoint_refresh])
        model.save_weights(os.path.join(log_dir, 'trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper_new(train_data_factory),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper_new(valid_data_factory),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, checkpoint_refresh, reduce_lr, early_stopping])
        model.save_weights(os.path.join(log_dir, 'trained_weights_final.h5'))

    train_data_factory.join()
    valid_data_factory.join()

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    backup_checkpoing = os.path.join(args.log_dir, 'refresh_checkpoint.h5')
    if os.path.exists(backup_checkpoing):
        weights_path = backup_checkpoing

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    backup_checkpoing = os.path.join(args.log_dir, 'refresh_checkpoint.h5')
    print(backup_checkpoing)
    if os.path.exists(backup_checkpoing):
        weights_path = backup_checkpoing

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        if (i+batch_size > n):
            np.random.shuffle(annotation_lines)
            i = 0
        output = threadpool.starmap(get_random_data, zip(annotation_lines[i:i+batch_size], itertools.repeat(input_shape, batch_size)))
        image_data = list(zip(*output))[0]
        box_data = list(zip(*output))[1]
        i = i+batch_size
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def data_generator_wrapper_new(data_factory):
    return data_factory.get_data()

class data_factory(threading.Thread):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        super().__init__()
        self.data_cond = threading.Condition()
        self.proc_cond = threading.Condition()
        self.output_datas = []
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        assert len(self.annotation_lines) > 0
        assert self.batch_size > 0

    def run(self):
        while True:
            cur_data_size = len(self.output_datas)
            if cur_data_size < 5:
                for i in range(cur_data_size, 10):
                    output_data = data_generator(self.annotation_lines, self.batch_size, self.input_shape, self.anchors, self.num_classes)
                    time.sleep(0.01)
                    self.data_cond.acquire()
                    self.output_datas.append(output_data)
                    self.data_cond.release()
            else:
                self.proc_cond.acquire()
                self.proc_cond.wait()
                self.proc_cond.release()

    def get_data(self):
        while True:
            Breakout=False
            self.data_cond.acquire()
            if len(self.output_datas) > 0:
                Breakout = True
                output_data = self.output_datas.pop()
                if len(self.output_datas) < 3:
                    self.proc_cond.acquire()
                    self.proc_cond.notify()
                    self.proc_cond.release()
            else:
                self.data_cond.wait()
            self.data_cond.release()
            if Breakout:
                return output_data

if __name__ == '__main__':
    _main()
