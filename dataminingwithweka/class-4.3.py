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

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.filters import Filter

jvm.start()

# load diabetes
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "diabetes.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
# we'll set the class attribute after filtering

# apply NominalToBinary filter and set class attribute
fltr = Filter("weka.filters.unsupervised.attribute.NominalToBinary")
fltr.inputformat(data)
filtered = fltr.filter(data)
filtered.class_is_last()

# cross-validate LinearRegression on filtered data, display model
cls = Classifier(classname="weka.classifiers.functions.LinearRegression")
pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
evl = Evaluation(filtered)
evl.crossvalidate_model(cls, filtered, 10, Random(1), pout)
print("10-fold cross-validation:\n" + evl.summary())
print("Predictions:\n\n" + str(pout))
cls.build_classifier(filtered)
print("Model:\n\n" + str(cls))

# use AddClassification filter with LinearRegression on filtered data
print("Applying AddClassification to filtered data:\n")
fltr = Filter(
    classname="weka.filters.supervised.attribute.AddClassification",
    options=["-W", "weka.classifiers.functions.LinearRegression", "-classification"])
fltr.inputformat(filtered)
classified = fltr.filter(filtered)
print(classified)

# convert class back to nominal
fltr = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "9"])
fltr.inputformat(classified)
nominal = fltr.filter(classified)

# delete attributes 1-8
fltr = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "1-8"])
fltr.inputformat(nominal)
twoatt = fltr.filter(nominal)
twoatt.class_index = twoatt.attribute_by_name("class").index

# cross-validate default OneR and output model
cls = Classifier(classname="weka.classifiers.rules.OneR")
evl = Evaluation(twoatt)
evl.crossvalidate_model(cls, twoatt, 10, Random(1))
print("10-fold cross-validation (default OneR):\n" + evl.summary())
cls.build_classifier(twoatt)
print("Model (default OneR):\n\n" + str(cls))

# cross-validate OneR with 100 buckets and output model
cls = Classifier(classname="weka.classifiers.rules.OneR", options=["-B", "100"])
evl = Evaluation(twoatt)
evl.crossvalidate_model(cls, twoatt, 10, Random(1))
print("10-fold cross-validation (OneR with 100 buckets):\n" + evl.summary())
cls.build_classifier(twoatt)
print("Model (OneR with 100 buckets):\n\n" + str(cls))

jvm.stop()
