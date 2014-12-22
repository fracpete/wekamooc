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

# More Data Mining with Weka - Class 4.2
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
from weka.classifiers import Classifier, Evaluation, SingleClassifierEnhancer
from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection

jvm.start()

# load glass
fname = data_dir + os.sep + "glass.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

classifiers = [
    "weka.classifiers.trees.J48",
    "weka.classifiers.lazy.IBk"
]

# cross-validate classifiers
for classifier in classifiers:
    cls = Classifier(classname=classifier)
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, 10, Random(1))
    print("%s: %0.0f%%" % (classifier, evl.percent_correct))

# wrapper
for classifier in classifiers:
    aseval = ASEvaluation(classname="weka.attributeSelection.WrapperSubsetEval",
                          options=["-B", classifier])
    assearch = ASSearch(classname="weka.attributeSelection.BestFirst",
                        options=[])
    attsel = AttributeSelection()
    attsel.evaluator(aseval)
    attsel.search(assearch)
    attsel.select_attributes(data)
    reduced = attsel.reduce_dimensionality(data)

    cls = Classifier(classname=classifier)
    evl = Evaluation(reduced)
    evl.crossvalidate_model(cls, reduced, 10, Random(1))
    print("%s (reduced): %0.0f%%" % (classifier, evl.percent_correct))

# meta-classifier
for wrappercls in classifiers:
    for basecls in classifiers:
        meta = SingleClassifierEnhancer(classname="weka.classifiers.meta.AttributeSelectedClassifier")
        meta.options = \
            ["-E", "weka.attributeSelection.WrapperSubsetEval -B " + wrappercls,
             "-S", "weka.attributeSelection.BestFirst",
             "-W", basecls]
        evl = Evaluation(data)
        evl.crossvalidate_model(meta, data, 10, Random(1))
        print("%s/%s: %0.0f%%" % (wrappercls, basecls, evl.percent_correct))

jvm.stop()
