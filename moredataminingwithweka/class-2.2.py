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

# More Data Mining with Weka - Class 2.2
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
from weka.filters import Filter
from weka.classifiers import Classifier, FilteredClassifier, Evaluation

jvm.start()


def get_nodes(s):
    """
    Obtains the nodes from J48's model string.
    :param s: the model string
    :type s: str
    :return: the number of nodes
    :rtype: str
    """
    result = "N/A"
    lines = s.split("\n")
    for line in lines:
        if line.find("Size of the tree :") > -1:
            result = line.replace("Size of the tree :", "").strip()
    return result


# load ionosphere
fname = data_dir + os.sep + "ionosphere.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# 1. cheating with default filter
fltr = Filter(classname="weka.filters.supervised.attribute.Discretize", options=[])
fltr.inputformat(data)
filtered = fltr.filter(data)
cls = Classifier(classname="weka.classifiers.trees.J48")
evl = Evaluation(filtered)
evl.crossvalidate_model(cls, filtered, 10, Random(1))
cls.build_classifier(filtered)
print("cheating (default): accuracy=%0.1f nodes=%s" % (evl.percent_correct, get_nodes(str(cls))))

# 2. using FilteredClassifier with default filter
cls = FilteredClassifier()
cls.classifier = Classifier(classname="weka.classifiers.trees.J48")
cls.filter = Filter(classname="weka.filters.supervised.attribute.Discretize", options=[])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
cls.build_classifier(data)
print("FilteredClassifier (default): accuracy=%0.1f nodes=%s" % (evl.percent_correct, get_nodes(str(cls))))

# 3. using FilteredClassifier (make binary)
cls = FilteredClassifier()
cls.classifier = Classifier(classname="weka.classifiers.trees.J48")
cls.filter = Filter(classname="weka.filters.supervised.attribute.Discretize", options=["-D"])
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
cls.build_classifier(data)
print("FilteredClassifier (make binary): accuracy=%0.1f nodes=%s" % (evl.percent_correct, get_nodes(str(cls))))

# 1. cheating with make binary
fltr = Filter(classname="weka.filters.supervised.attribute.Discretize", options=["-D"])
fltr.inputformat(data)
filtered = fltr.filter(data)
cls = Classifier(classname="weka.classifiers.trees.J48")
evl = Evaluation(filtered)
evl.crossvalidate_model(cls, filtered, 10, Random(1))
cls.build_classifier(filtered)
print("cheating (make binary): accuracy=%0.1f nodes=%s" % (evl.percent_correct, get_nodes(str(cls))))

jvm.stop()
