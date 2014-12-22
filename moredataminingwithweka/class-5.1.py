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

# More Data Mining with Weka - Class 5.1
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
from weka.classifiers import Classifier, Evaluation

jvm.start()

datasets = [
    "ionosphere.arff",
    "credit-g.arff",
    "breast-cancer.arff",
    "diabetes.arff"
]
classifiers = [
    "weka.classifiers.functions.VotedPerceptron",
    "weka.classifiers.functions.SMO",
]

for dataset in datasets:
    # load dataset
    fname = data_dir + os.sep + dataset
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(fname)
    data.class_is_last()

    for classifier in classifiers:
        # cross-validate classifier
        cls = Classifier(classname=classifier)
        evl = Evaluation(data)
        evl.crossvalidate_model(cls, data, 10, Random(1))
        print("%s / %s: %0.1f%%" % (dataset, classifier, evl.percent_correct))

jvm.stop()
