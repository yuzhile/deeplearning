#!/usr/bin/env python
# encoding: utf-8

import cPickle,gzip,numpy
f = gzip.open('mnist.pkl.gz','rb')
train_set,valid_set,test_set = cPickle.load(f)
train_x,train_y = train_set
print train_x
print train_y
print numpy.distutils.__config__.show()
f.close()
