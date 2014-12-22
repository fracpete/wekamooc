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

# More Data Mining with Weka - Class 4.4
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

jvm.start()

# load ionosphere
fname = data_dir + os.sep + "ionosphere.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

classifiers = [
    "weka.classifiers.bayes.NaiveBayes",
    "weka.classifiers.lazy.IBk",
    "weka.classifiers.trees.J48"
]

# cross-validate classifiers
for classifier in classifiers:
    # classifier itself
    cls = Classifier(classname=classifier)
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, 10, Random(1))
    print("%s: %0.0f%%" % (classifier, evl.percent_correct))
    # meta with cfssubseteval
    meta = SingleClassifierEnhancer(classname="weka.classifiers.meta.AttributeSelectedClassifier")
    meta.options = \
        ["-E", "weka.attributeSelection.CfsSubsetEval",
         "-S", "weka.attributeSelection.BestFirst",
         "-W", classifier]
    evl = Evaluation(data)
    evl.crossvalidate_model(meta, data, 10, Random(1))
    print("%s (cfs): %0.0f%%" % (classifier, evl.percent_correct))
    # meta with wrapper
    meta = SingleClassifierEnhancer(classname="weka.classifiers.meta.AttributeSelectedClassifier")
    meta.options = \
        ["-E", "weka.attributeSelection.WrapperSubsetEval -B " + classifier,
         "-S", "weka.attributeSelection.BestFirst",
         "-W", classifier]
    evl = Evaluation(data)
    evl.crossvalidate_model(meta, data, 10, Random(1))
    print("%s (wrapper): %0.0f%%" % (classifier, evl.percent_correct))
    # meta with gainratio
    meta = SingleClassifierEnhancer(classname="weka.classifiers.meta.AttributeSelectedClassifier")
    meta.options = \
        ["-E", "weka.attributeSelection.GainRatioAttributeEval",
         "-S", "weka.attributeSelection.Ranker -N 7",
         "-W", classifier]
    evl = Evaluation(data)
    evl.crossvalidate_model(meta, data, 10, Random(1))
    print("%s (gain ratio): %0.0f%%" % (classifier, evl.percent_correct))

jvm.stop()
