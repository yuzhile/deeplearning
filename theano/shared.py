#!/usr/bin/env python
# encoding: utf-8

from theano import shared
import theano.tensor as T
from theano import function
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc],state,updates=[(state,state+inc)])
print accumulator(2)
print accumulator(2)
print accumulator(2)
print accumulator(2)

