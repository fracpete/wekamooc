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

# More Data Mining with Weka - Class 4.6
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

from numpy import array
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation, CostMatrix
import weka.classifiers as classifiers

jvm.start()

# load credit-g
fname = data_dir + os.sep + "credit-g.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# cross-validate NaiveBayes
classifier = "weka.classifiers.bayes.NaiveBayes"
print("\n--> " + classifier + "\n")
cls = Classifier(classname=classifier)
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
preds = classifiers.predictions_to_instances(data, evl.predictions)
preds.sort(preds.attribute_by_name("distribution-good").index)
print(evl.summary())
print(evl.matrix())
print(preds)

# cross-validate J48
classifier = "weka.classifiers.trees.J48"
print("\n--> " + classifier + "\n")
cls = Classifier(classname=classifier, options=["-M", "100"])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
preds = classifiers.predictions_to_instances(data, evl.predictions)
preds.sort(preds.attribute_by_name("distribution-good").index)
print(evl.summary())
print(evl.matrix())
print(preds)

# cross-validate CostSensitiveClassifier with J48 (minimize cost)
classifier = "weka.classifiers.meta.CostSensitiveClassifier"
base = "weka.classifiers.trees.J48"
print("\n--> " + classifier + "/" + base + "\n")
cost = array([[0, 1], [5, 0]])
matrx = CostMatrix(matrx=cost)
cls = Classifier(classname=classifier,
                 options=["-M", "-W", base, "-cost-matrix", matrx.to_matlab()])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("Accuracy: %0.1f" % evl.percent_correct)
print(evl.matrix())

# cross-validate Bagging with J48
classifier = "weka.classifiers.meta.Bagging"
base = "weka.classifiers.trees.J48"
print("\n--> " + classifier + "/" + base + "\n")
cls = Classifier(classname=classifier,
                 options=["-W", base])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("Accuracy: %0.1f" % evl.percent_correct)
print(evl.matrix())

# cross-validate CostSensitiveClassifier with NaiveBayes
classifier = "weka.classifiers.meta.CostSensitiveClassifier"
base = "weka.classifiers.bayes.NaiveBayes"
print("\n--> " + classifier + "/" + base + "\n")
cost = array([[0, 1], [5, 0]])
matrx = CostMatrix(matrx=cost)
cls = Classifier(classname=classifier,
                 options=["-W", base, "-cost-matrix", matrx.to_matlab()])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("Accuracy: %0.1f" % evl.percent_correct)
print(evl.matrix())

# cross-validate CostSensitiveClassifier with J48
classifier = "weka.classifiers.meta.CostSensitiveClassifier"
base = "weka.classifiers.trees.J48"
print("\n--> " + classifier + "/" + base + "\n")
cost = array([[0, 1], [5, 0]])
matrx = CostMatrix(matrx=cost)
cls = Classifier(classname=classifier,
                 options=["-W", base, "-cost-matrix", matrx.to_matlab()])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("Accuracy: %0.1f" % evl.percent_correct)
print(evl.matrix())

# cross-validate CostSensitiveClassifier with Bagged J48
classifier = "weka.classifiers.meta.CostSensitiveClassifier"
base = "weka.classifiers.meta.Bagging"
print("\n--> " + classifier + "/" + base + "\n")
print("\n--> " + classifier + "\n")
cost = array([[0, 1], [5, 0]])
matrx = CostMatrix(matrx=cost)
cls = Classifier(classname=classifier,
                 options=["-W", base, "-cost-matrix", matrx.to_matlab(), "--", "-W", "weka.classifiers.trees.J48"])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("Accuracy: %0.1f" % evl.percent_correct)
print(evl.matrix())

jvm.stop()
