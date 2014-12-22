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

# Data Mining with Weka - Class 1.6
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
import weka.plot.dataset as pld
import weka.plot.classifiers as plc

jvm.start()

# load iris
fname = data_dir + os.sep + "iris.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# plot
pld.scatter_plot(
    data, data.attribute_by_name("petalwidth").index,
    data.attribute_by_name("petallength").index,
    wait=False)

# add classifier errors to dataset
addcls = Filter(
    classname="weka.filters.supervised.attribute.AddClassification",
    options=["-W", "weka.classifiers.trees.J48", "-classification", "-error"])
addcls.inputformat(data)
filtered = addcls.filter(data)
print(filtered)

# build J48
cls = Classifier(classname="weka.classifiers.trees.J48")
cls.build_classifier(data)
evl = Evaluation(data)
evl.test_model(cls, data)

# plot classifier errors
plc.plot_classifier_errors(evl.predictions, wait=True)

jvm.stop()

