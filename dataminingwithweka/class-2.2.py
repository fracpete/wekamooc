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

# Data Mining with Weka - Class 2.2
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

import os
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation

jvm.start()

# TODO
# wherever your datasets are located
data_dir = "/some/where/data"

# load segment-challenge
loader = Loader(classname="weka.core.converters.ArffLoader")

fname = data_dir + os.sep + "segment-challenge.arff"
print("\nLoading dataset: " + fname + "\n")
train = loader.load_file(fname)
train.set_class_index(train.num_attributes() - 1)

fname = data_dir + os.sep + "segment-test.arff"
print("\nLoading dataset: " + fname + "\n")
test = loader.load_file(fname)
test.set_class_index(train.num_attributes() - 1)

# build J48
cls = Classifier(classname="weka.classifiers.trees.J48")
cls.build_classifier(train)

# evaluate on test
evl = Evaluation(train)
evl.test_model(cls, test)
print("Test set accuracy: %0.0f%%" % evl.percent_correct())

# evaluate on train
evl = Evaluation(train)
evl.test_model(cls, train)
print("Train set accuracy: %0.0f%%" % evl.percent_correct())

# evaluate on random split
evl = Evaluation(train)
evl.evaluate_train_test_split(cls, train, 66.0, Random(1))
print("Random split accuracy: %0.0f%%" % evl.percent_correct())

jvm.stop()
