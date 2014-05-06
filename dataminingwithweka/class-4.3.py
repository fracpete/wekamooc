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

# Data Mining with Weka - Class 4.3
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# TODO
# wherever your datasets are located
#data_dir = "/some/where/data"
data_dir = "/home/fracpete/development/projects/wekamooc/dataminingwithweka/data/"

import os
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.filters import Filter
import weka.plot.graph as plg

jvm.start()

# load diabetes
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "diabetes.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
# we'll set the class attribute after filtering

# apply NominalToBinary filter and set class attribute
fltr = Filter("weka.filters.unsupervised.attribute.NominalToBinary")
fltr.set_inputformat(data)
filtered = fltr.filter(data)
filtered.set_class_index(data.num_attributes() - 1)

# cross-validate LinearRegression on filtered data, display model
cls = Classifier(classname="weka.classifiers.functions.LinearRegression")
pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
evl = Evaluation(filtered)
evl.crossvalidate_model(cls, filtered, 10, Random(1), pout)
print("10-fold cross-validation:\n" + evl.to_summary())
print("Predictions:\n\n" + pout.get_buffer_content())
cls.build_classifier(filtered)
print("Model:\n\n" + str(cls))

# use AddClassification filter with LinearRegression on filtered data
print("Applying AddClassification to filtered data:\n")
fltr = Filter(
    classname="weka.filters.supervised.attribute.AddClassification",
    options=["-W", "weka.classifiers.functions.LinearRegression", "-classification"])
fltr.set_inputformat(filtered)
classified = fltr.filter(filtered)
print(classified)

# convert class back to nominal
fltr = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "9"])
fltr.set_inputformat(classified)
nominal = fltr.filter(classified)

# delete attributes 1-8
fltr = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "1-8"])
fltr.set_inputformat(nominal)
twoatt = fltr.filter(nominal)
twoatt.set_class_index(twoatt.get_attribute_by_name("class").get_index())

# cross-validate default OneR and output model
cls = Classifier(classname="weka.classifiers.rules.OneR")
evl = Evaluation(twoatt)
evl.crossvalidate_model(cls, twoatt, 10, Random(1), pout)
print("10-fold cross-validation (default OneR):\n" + evl.to_summary())
cls.build_classifier(twoatt)
print("Model (default OneR):\n\n" + str(cls))

# cross-validate OneR with 100 buckets and output model
cls = Classifier(classname="weka.classifiers.rules.OneR", options=["-B", "100"])
evl = Evaluation(twoatt)
evl.crossvalidate_model(cls, twoatt, 10, Random(1))
print("10-fold cross-validation (OneR with 100 buckets):\n" + evl.to_summary())
cls.build_classifier(twoatt)
print("Model (OneR with 100 buckets):\n\n" + str(cls))

jvm.stop()
