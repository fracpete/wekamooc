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

# Data Mining with Weka - Class 2.4
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

import os
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation

jvm.start()

# TODO
# wherever your datasets are located
data_dir = "/some/where/data"

classifiers = [
    "weka.classifiers.rules.ZeroR",
    "weka.classifiers.trees.J48",
    "weka.classifiers.bayes.NaiveBayes",
    "weka.classifiers.lazy.IBk",
    "weka.classifiers.rules.PART"
]

# load diabetes
data = data_dir + os.sep + "diabetes.arff"
print("\nLoading dataset: " + data + "\n")
loader = Loader("weka.core.converters.ArffLoader")
data = loader.load_file(data)
data.set_class_index(data.num_attributes() - 1)

# random split on data
for classifier in classifiers:
    cls = Classifier(classname=classifier)
    evl = Evaluation(data)
    evl.evaluate_train_test_split(cls, data, 66.0, Random(1))
    print(classifier + ": %0.0f" % evl.percent_correct())

# load supermarket
data = data_dir + os.sep + "supermarket.arff"
print("\nLoading dataset: " + data + "\n")
loader = Loader("weka.core.converters.ArffLoader")
data = loader.load_file(data)
data.set_class_index(data.num_attributes() - 1)

# random split on data
for classifier in classifiers:
    cls = Classifier(classname=classifier)
    evl = Evaluation(data)
    evl.evaluate_train_test_split(cls, data, 66.0, Random(1))
    print(classifier + ": %0.0f" % evl.percent_correct())

jvm.stop()

