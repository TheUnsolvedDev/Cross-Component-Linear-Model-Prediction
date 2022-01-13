import os,sys,random,math
from wsgiref import validate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import *
from dataset import *

if __name__ == '__main__':
    data = read_data_sets()
    train = data.train
    test = data.test
    validation = data.validation
    
    