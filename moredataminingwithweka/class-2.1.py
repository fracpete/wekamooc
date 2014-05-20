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

# More Data Mining with Weka - Class 2.1
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# TODO
# wherever your datasets are located
data_dir = "/some/where/data"

import os
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.filters import Filter
from weka.classifiers import Classifier, Evaluation

jvm.start()

# load ionosphere
fname = data_dir + os.sep + "ionosphere.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.set_class_index(data.num_attributes() - 1)

for equal in ["", "-F"]:
    print("\nEqual frequency binning? " + str(equal == "-F") + "\n")
    for bins in [0, 40, 10, 5, 2]:
        if bins > 0:
            fltr = Filter(classname="weka.filters.unsupervised.attribute.Discretize", options=["-B", str(bins), equal])
            fltr.set_inputformat(data)
            filtered = fltr.filter(data)
        else:
            filtered = data
        cls = Classifier("weka.classifiers.trees.J48")
        # cross-validate
        evl = Evaluation(filtered)
        evl.crossvalidate_model(cls, filtered, 10, Random(1))
        # build classifier on full dataset
        cls.build_classifier(filtered)
        # get size of tree from model strings
        lines = str(cls).split("\n")
        nodes = "N/A"
        for line in lines:
            if line.find("Size of the tree :") > -1:
                nodes = line.replace("Size of the tree :", "").strip()
        # output stats
        print("bins=%i accuracy=%0.1f nodes=%s" % (bins, evl.percent_correct(), nodes))

jvm.stop()
