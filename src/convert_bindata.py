from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import cv2
import h5py
import numpy as np
import math
import random
import logging
import pickle
from data import FaceImageIter
from data import FaceImageIterList
import mxnet as mx
from mxnet import ndarray as nd
from datetime import datetime
import time
#sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'common'))
from common import face_image
import reco2.common as cmn

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-path', default='', help='')
  parser.add_argument('--dst-path', type=str, default='.', help='')
  parser.add_argument('--split', type=float, default=0.8, help='train/test split')
  parser.add_argument('--rescale_size', type=int, default=128, help='rescale images to this square size')
  parser.add_argument('--mean', type=float, default=127.7, help='subtract mean')
  parser.add_argument('--norm_scale', type=float, default=0.0078125, help='scale image after mean subtraction')
  parser.add_argument('--chunks', type=int, default=3, help='number of chunks to split the data to')
  args = parser.parse_args()
  return args

def convert_data(args):
  path_imgrec = os.path.join(args.data_path, 'train.rec')
  path_imgidx = os.path.join(args.data_path, 'train.idx')
  imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
  s = imgrec.read_idx(0)
  header, _ = mx.recordio.unpack(s)
  assert(header.flag>0)
  print('header0 label', header.label)
  header0 = (int(header.label[0]), int(header.label[1]))
  print('identities', header0[1]-header0[0], 'images', header0[0])

  id2range = {}
  seq_identity = range(int(header.label[0]), int(header.label[1]))
  for identity in seq_identity:
    s = imgrec.read_idx(identity)
    header, _ = mx.recordio.unpack(s)
    id2range[identity] = (int(header.label[0]), int(header.label[1]))
  print('id2range', len(id2range))
  prop = face_image.load_property(args.data_path)
  image_size = prop.image_size
  print('image_size', image_size)
  #for _id, v in id2range.iteritems():

  pid = 0
  for identity in seq_identity:
    s = imgrec.read_idx(identity)
    header, _ = mx.recordio.unpack(s)
    print(header)

    pid +=1 

    outpath = os.path.join(args.dst_path, str(header.id))
    if not os.path.exists(outpath):
      os.makedirs(outpath)

    a, b = int(header.label[0]), int(header.label[1])
    for _idx in range(a,b):
      s = imgrec.read_idx(_idx)
      _header, _content = mx.recordio.unpack(s)
      bgr = mx.image.imdecode(_content, flag=0)
      bgr = mx.image.resize_short(bgr, args.rescale_size)
      img = bgr.asnumpy()[...,::-1]
      cv2.imwrite(os.path.join(outpath, "{}_{}.png".format(header.id, _idx)), img)

      info_path = os.path.join(outpath, 'info.txt')
      if not os.path.exists(info_path):
        dtt = datetime.now()
        dt =time.strftime("%Y%m%dT%H%M%S")# + str(dtt.microsecond)
        experiment = 'enroll'
        tags = ''
        templateuid = str(header.id)

        info = cmn.InfoTxt(pid=pid, info=str(header.id), datatype=cmn.RECO2_DATA_TYPE.IMG, datetime = dt, schemeid='MSCeleb', experiment=experiment, tags = tags, templateuid=templateuid)
        info.save(outpath)

  print('done.')


def convert_data_2hdf5(args):
  path_imgrec = os.path.join(args.data_path, 'train.rec')
  path_imgidx = os.path.join(args.data_path, 'train.idx')
  imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
  s = imgrec.read_idx(0)
  header, _ = mx.recordio.unpack(s)
  assert(header.flag>0)
  print('header0 label', header.label)
  header0 = (int(header.label[0]), int(header.label[1]))
  n_ids = header0[1]-header0[0]
  print('identities',n_ids, 'images', header0[0])

  # id2range = {}
  # seq_identity = range(int(header.label[0]), int(header.label[1]))
  # for identity in seq_identity:
  #   s = imgrec.read_idx(identity)
  #   header, _ = mx.recordio.unpack(s)
  #   id2range[identity] = (int(header.label[0]), int(header.label[1]))
  # print('id2range', len(id2range))
  prop = face_image.load_property(args.data_path)
  image_size = prop.image_size
  print('image_size', image_size)

  if not os.path.exists(args.dst_path):
    os.makedirs(args.dst_path)


  pid = 0
  dataTrain = []
  labelsTrain = []
  dataTest = []
  labelsTest = []

  n_chunk = 0
  start = int(header.label[0])
  stop = int(header.label[1])
  #stop = start + 10
  splits = np.array_split(range(start, stop), args.chunks)
  for split in splits:
    for identity in split:
      s = imgrec.read_idx(identity)
      header, _ = mx.recordio.unpack(s)
      pid +=1 
      print(header, pid)

      a, b = int(header.label[0]), int(header.label[1])
      aTrain, bTrain = a, int(args.split * b + a * (1 - args.split))
      for _idx in range(aTrain, bTrain):
        s = imgrec.read_idx(_idx)
        _header, _content = mx.recordio.unpack(s)
        img = mx.image.imdecode(_content, flag=0)  #flag=0 means grayscale image, flag=1 means bgr
        img = prep_image(img)
        dataTrain.append(img)
        labelsTrain.append(pid)

      aTest, bTest = bTrain, b
      for _idx in range(aTest, bTest):
        s = imgrec.read_idx(_idx)
        _header, _content = mx.recordio.unpack(s)
        img = mx.image.imdecode(_content, flag=0)
        img = prep_image(img)
        dataTest.append(img)
        labelsTest.append(pid)

    to_hdf5(os.path.join(args.dst_path, 'train_chunk' + str(n_chunk) + '.h5'), dataTrain, labelsTrain)
    to_hdf5(os.path.join(args.dst_path, 'test_chunk' + str(n_chunk) + '.h5'), dataTest, labelsTest)

    dataTrain = []
    labelsTrain = []
    dataTest = []
    labelsTest = []
    n_chunk += 1

def to_hdf5(dst, data, labels):
  print('saving to', dst)
  #reshape images to 4-D: [col, rows, channel,numbers]
  nsamples = len(labels)
  data = np.asarray(data)
  labels = np.asarray(labels)
  data = data.reshape((nsamples, 1, args.rescale_size, args.rescale_size))
  labels = labels.reshape((nsamples, 1))

  print('store the data and label to', dst)
  with h5py.File(dst, 'w-') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=labels)
  hf.close()

def prep_image(img):
  img = mx.image.resize_short(img, args.rescale_size)
  img = img.asnumpy()[...,::-1]
  #img = args.norm_scale * (img - args.mean)
  return img

def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    #convert_data(args)
    convert_data_2hdf5(args)

if __name__ == '__main__':
    main()

