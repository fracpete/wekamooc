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

# More Data Mining with Weka - Class 2.6
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import FilteredClassifier, Classifier, Evaluation
from weka.filters import Filter
import weka.plot.classifiers as plc

jvm.start()

# load ReutersGrain-train
fname = data_dir + os.sep + "ReutersGrain-train.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# load ReutersGrain-train
fname = data_dir + os.sep + "ReutersGrain-test.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
test = loader.load_file(fname)
test.class_is_last()

setups = (
    ("weka.classifiers.trees.J48", []),
    ("weka.classifiers.bayes.NaiveBayes", []),
    ("weka.classifiers.bayes.NaiveBayesMultinomial", []),
    ("weka.classifiers.bayes.NaiveBayesMultinomial", ["-C"]),
    ("weka.classifiers.bayes.NaiveBayesMultinomial", ["-C", "-L", "-stopwords-handler", "weka.core.stopwords.Rainbow"])
)

# cross-validate classifiers
for setup in setups:
    classifier, opt = setup
    print("\n--> %s (filter options: %s)\n" % (classifier, " ".join(opt)))
    cls = FilteredClassifier()
    cls.classifier = Classifier(classname=classifier)
    cls.filter = Filter(classname="weka.filters.unsupervised.attribute.StringToWordVector", options=opt)
    cls.build_classifier(data)
    evl = Evaluation(test)
    evl.test_model(cls, test)
    print("Accuracy: %0.0f%%" % evl.percent_correct)
    tcdata = plc.generate_thresholdcurve_data(evl, 0)
    print("AUC: %0.3f" % plc.get_auc(tcdata))
    print(evl.matrix("Matrix:"))

jvm.stop()
