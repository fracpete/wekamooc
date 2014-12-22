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

# Data Mining with Weka - Class 4.4
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

for classifier in ["weka.classifiers.bayes.NaiveBayes", "weka.classifiers.rules.ZeroR", "weka.classifiers.trees.J48"]:
    # train/test split 90% using classifier
    cls = Classifier(classname=classifier)
    evl = Evaluation(data)
    evl.evaluate_train_test_split(cls, data, 90.0, Random(1))
    print("\n" + classifier + " train/test split (90%):\n" + evl.summary())
    cls.build_classifier(data)
    print(classifier + " model:\n\n" + str(cls))

# calculate mean/stdev over 10 cross-validations
for classifier in [
    "weka.classifiers.meta.ClassificationViaRegression", "weka.classifiers.bayes.NaiveBayes",
        "weka.classifiers.rules.ZeroR", "weka.classifiers.trees.J48", "weka.classifiers.functions.Logistic"]:
    accuracy = []
    for i in xrange(1,11):
        cls = Classifier(classname=classifier)
        evl = Evaluation(data)
        evl.crossvalidate_model(cls, data, 10, Random(i))
        accuracy.append(evl.percent_correct)
    nacc = numpy.array(accuracy)
    print("%s: %0.2f +/-%0.2f" % (classifier, numpy.mean(nacc), numpy.std(nacc)))

jvm.stop()
