# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Data Mining with Weka - Class 2.3
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import numpy
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation

jvm.start()

# load segment-challenge
loader = Loader(classname="weka.core.converters.ArffLoader")

fname = data_dir + os.sep + "segment-challenge.arff"
print("\nLoading dataset: " + fname + "\n")
train = loader.load_file(fname)
train.class_is_last()

# use seed 1-10 and perform random split with 90%
perc = []
for i in xrange(1, 11):
    evl = Evaluation(train)
    evl.evaluate_train_test_split(
        Classifier(classname="weka.classifiers.trees.J48"), train, 90.0, Random(i))
    perc.append(round(evl.percent_correct, 1))
    print("Accuracy with seed %i: %0.1f%%" % (i, evl.percent_correct))

# calculate mean and standard deviation
nperc = numpy.array(perc)
print("mean=%0.2f stdev=%0.2f" % (numpy.mean(nperc), numpy.std(nperc)))

jvm.stop()
