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

# Data Mining with Weka - Class 5.2
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
from weka.core.dataset import Instances
from weka.classifiers import Classifier, Evaluation

jvm.start()

# load weather.nominal
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "weather.nominal.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.class_is_last()

# define classifiers
classifiers = ["weka.classifiers.rules.OneR", "weka.classifiers.trees.J48"]

# cross-validate original dataset
for classifier in classifiers:
    cls = Classifier(classname=classifier)
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, 10, Random(1))
    print("%s (original): %0.0f%%" % (classifier, evl.percent_correct))

# replace 'outlook' in first 4 'no' instances with 'missing'
modified = Instances.copy_instances(data)
count = 0
for i in xrange(modified.num_instances):
    if modified.get_instance(i).get_string_value(modified.class_index) == "no":
        count += 1
        modified.get_instance(i).set_missing(0)
        if count == 4:
            break

# cross-validate modified dataset
for classifier in classifiers:
    cls = Classifier(classname=classifier)
    evl = Evaluation(modified)
    evl.crossvalidate_model(cls, modified, 10, Random(1))
    print("%s (modified): %0.0f%%" % (classifier, evl.percent_correct))

jvm.stop()
