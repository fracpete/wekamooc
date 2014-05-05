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

# Data Mining with Weka - Class 2.5
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

import os
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter

jvm.start()

# TODO
# wherever your datasets are located
data_dir = "/some/where/data"

# load segment-challenge
loader = Loader(classname="weka.core.converters.ArffLoader")

fname = data_dir + os.sep + "diabetes.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.set_class_index(data.num_attributes() - 1)

# simulate the 10 train/test pairs of cross-validation
evl = Evaluation(data)
for i in xrange(1, 11):
    # create train set
    remove = Filter(
        classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
        options=["-N", "10", "-F", str(i), "-S", "1", "-V"])
    remove.set_inputformat(data)
    train = remove.filter(data)

    # create test set
    remove = Filter(
        classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
        options=["-N", "10", "-F", str(i), "-S", "1"])
    remove.set_inputformat(data)
    test = remove.filter(data)

    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    evl.test_model(cls, test)

print("Simulated CV accuracy: %0.1f%%" % (evl.percent_correct()))

# perform actual cross-validation
evl = Evaluation(data)
cls = Classifier(classname="weka.classifiers.trees.J48")
evl.crossvalidate_model(cls, data, 10, Random(1))

print("Actual CV accuracy: %0.1f%%" % (evl.percent_correct()))

# deploy
print("Build model on full dataset:\n")
cls = Classifier(classname="weka.classifiers.trees.J48")
cls.build_classifier(data)
print(cls)

jvm.stop()
