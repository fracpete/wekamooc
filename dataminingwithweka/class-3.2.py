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

# Data Mining with Weka - Class 3.1
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

# load weather.nominal
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "weather.nominal.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.set_class_index(data.num_attributes() - 1)

# perform 10-fold cross-validation
cls = Classifier(classname="weka.classifiers.rules.OneR")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("10-fold cross-validation (full):\n" + evl.to_summary())
cls.build_classifier(data)
print("Model:\n\n" + str(cls))

# remove attribute "outlook"
print("Removing attribute 'outlook'")
data.delete_attribute(data.get_attribute_by_name("outlook").get_index())

# perform 10-fold cross-validation (reduced dataset)
cls = Classifier(classname="weka.classifiers.rules.OneR")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("10-fold cross-validation (without 'outlook'):\n" + evl.to_summary())
cls.build_classifier(data)
print("Model:\n\n" + str(cls))

# load diabetes
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "diabetes.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.set_class_index(data.num_attributes() - 1)

cls = Classifier(classname="weka.classifiers.rules.ZeroR")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("Accuracy 10-fold cross-validation (ZeroR): %0.1f%%" % evl.percent_correct())

cls = Classifier(classname="weka.classifiers.rules.OneR")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("Accuracy 10-fold cross-validation (OneR): %0.1f%%" % evl.percent_correct())
cls.build_classifier(data)
print(cls)

cls = Classifier(classname="weka.classifiers.rules.OneR", options=["-B", "1"])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("Accuracy 10-fold cross-validation (OneR -B 1): %0.1f%%" % evl.percent_correct())

cls = Classifier(classname="weka.classifiers.rules.OneR", options=["-B", "1"])
cls.build_classifier(data)
evl = Evaluation(data)
evl.test_model(cls, data)
print("Accuracy on training data (OneR -B 1): %0.1f%%" % evl.percent_correct())
print(cls)

jvm.stop()
