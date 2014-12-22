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

# Data Mining with Weka - Class 2.6
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

# load diabetes
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "diabetes.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.class_is_last()

# determine baseline with ZeroR
zeror = Classifier(classname="weka.classifiers.rules.ZeroR")
zeror.build_classifier(data)
evl = Evaluation(data)
evl.test_model(zeror, data)
print("Baseline accuracy (ZeroR): %0.1f%%" % evl.percent_correct)

print("\nHoldout 10%...")
# use seed 1-10 and perform random split with 90%
perc = []
for i in xrange(1, 11):
    evl = Evaluation(data)
    evl.evaluate_train_test_split(
        Classifier(classname="weka.classifiers.trees.J48"), data, 90.0, Random(i))
    perc.append(round(evl.percent_correct, 1))
    print("Accuracy with seed %i: %0.1f%%" % (i, evl.percent_correct))

# calculate mean and standard deviation
nperc = numpy.array(perc)
print("mean=%0.2f stdev=%0.2f" % (numpy.mean(nperc), numpy.std(nperc)))

print("\n10-fold Cross-validation...")
# use seed 1-10 and perform 10-fold CV
perc = []
for i in xrange(1, 11):
    evl = Evaluation(data)
    evl.crossvalidate_model(Classifier(classname="weka.classifiers.trees.J48"), data, 10, Random(i))
    perc.append(round(evl.percent_correct, 1))
    print("Accuracy with seed %i: %0.1f%%" % (i, evl.percent_correct))

# calculate mean and standard deviation
nperc = numpy.array(perc)
print("mean=%0.2f stdev=%0.2f" % (numpy.mean(nperc), numpy.std(nperc)))

jvm.stop()
