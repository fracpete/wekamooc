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
import numpy
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter

jvm.start()

# TODO
# wherever your datasets are located
data_dir = "/some/where/data"

# load segment-challenge
loader = Loader(classname="weka.core.converters.ArffLoader")

fname = data_dir + os.sep + "weather.nominal.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.set_class_index(data.num_attributes() - 1)

# perform 10-fold cross-validation
cls = Classifier(classname="weka.classifiers.rules.OneR")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("10-fold cross-validation:\n" + evl.to_summary())

# build model on full dataset and output it
cls.build_classifier(data)
print("Model:\n\n" + str(cls))

jvm.stop()
