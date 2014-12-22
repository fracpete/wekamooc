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

# More Data Mining with Weka - Class 5.4
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import weka.core.jvm as jvm
import weka.core.packages as packages
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation

jvm.start(packages=True)

# install GridSearch package
pkgs = ["gridSearch", "partialLeastSquares", "thresholdSelector"]
installed = True
for pkg in pkgs:
    if not packages.is_installed(pkg):
        installed = False
        print("Installing " + pkg)
        packages.install_package(pkg)
if not installed:
    print("Installed package(s), please restart")
    jvm.stop()
    exit()

# load diabetes
fname = data_dir + os.sep + "diabetes.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# J48
cls = Classifier(classname="weka.classifiers.trees.J48")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("J48: %0.1f%%" % evl.percent_correct)

# CVParameterSelection with J48 - confidenceFactor
cls = Classifier(classname="weka.classifiers.meta.CVParameterSelection",
                 options=["-W", "weka.classifiers.trees.J48", "-P", "C 0.1 0.9 9"])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("CVParameterSelection (confidenceFactor): %0.1f%%" % evl.percent_correct)

# CVParameterSelection with J48 - confidenceFactor+minNumObj
cls = Classifier(classname="weka.classifiers.meta.CVParameterSelection",
                 options=["-W", "weka.classifiers.trees.J48", "-P", "C 0.1 0.9 9", "-P", "M 1 10 10"])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("CVParameterSelection (confidenceFactor+minNumObj): %0.1f%%" % evl.percent_correct)

# GridSearch with J48 - confidenceFactor+minNumObj
cls = Classifier(classname="weka.classifiers.meta.GridSearch",
                 options=[
                     "-E", "ACC",
                     "-W", "weka.classifiers.trees.J48",
                     "-filter", "weka.filters.AllFilter",
                     "-x-property", "classifier.confidenceFactor",
                     "-x-min", "0.1",
                     "-x-max", "1.0",
                     "-x-step", "0.1",
                     "-x-expression", "I",
                     "-y-property", "classifier.minNumObj",
                     "-y-min", "1",
                     "-y-max", "10",
                     "-y-step", "1",
                     "-y-expression", "I"
                 ])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("GridSearch (confidenceFactor+minNumObj): %0.1f%%" % evl.percent_correct)

# load credit-g
fname = data_dir + os.sep + "credit-g.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# NaiveBayes
cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("NaiveBayes: %0.1f%%" % evl.percent_correct)

# ThresholdSelector with NaiveBayes
cls = Classifier(classname="weka.classifiers.meta.ThresholdSelector",
                 options=["-W", "weka.classifiers.bayes.NaiveBayes", "-M", "ACCURACY", "-E", "0", "-C", "1"])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("ThresholdSelector: %0.1f%%" % evl.percent_correct)

jvm.stop()
