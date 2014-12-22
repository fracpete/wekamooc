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

# Data Mining with Weka - Class 4.6
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
import weka.core.packages as packages
from weka.classifiers import Classifier, Evaluation

jvm.start(packages=True)

# install stackingC if necessary
if not packages.is_installed("stackingC"):
    print("Installing stackingC...")
    packages.install_package("stackingC")
    jvm.stop()
    print("Installed package, please restart")
    exit()

# load glass
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "glass.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.class_is_last()

# compare several meta-classifiers with J48
for classifier in [("weka.classifiers.trees.J48", []), ("weka.classifiers.meta.Bagging", []),
                   ("weka.classifiers.trees.RandomForest", []), ("weka.classifiers.meta.AdaBoostM1", []),
                   ("weka.classifiers.meta.Stacking", []),
                   ("weka.classifiers.meta.StackingC", ["-B", "weka.classifiers.lazy.IBk", "-B", "weka.classifiers.rules.PART", "-B", "weka.classifiers.trees.J48"])]:

    # cross-validate classifier
    cname, coptions = classifier
    cls = Classifier(classname=cname, options=coptions)
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, 10, Random(1))
    print(cname + " cross-validated accuracy: %0.2f" % evl.percent_correct)

jvm.stop()
