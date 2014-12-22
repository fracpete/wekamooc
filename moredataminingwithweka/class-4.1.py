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

# More Data Mining with Weka - Class 4.1
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
from weka.filters import Filter
from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection

jvm.start()

# load glass
fname = data_dir + os.sep + "glass.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# cross-validate J48
cls = Classifier(classname="weka.classifiers.trees.J48")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("All attributes: %0.0f%%" % evl.percent_correct)

# remove attributes (1) and cross-validate J48
atts = "RI|Mg|Type"
flt = Filter(classname="weka.filters.unsupervised.attribute.RemoveByName", options=["-E", "(" + atts + ")", "-V"])
flt.inputformat(data)
filtered = flt.filter(data)
cls = Classifier(classname="weka.classifiers.trees.J48")
evl = Evaluation(filtered)
evl.crossvalidate_model(cls, filtered, 10, Random(1))
print(atts + ": %0.0f%%" % evl.percent_correct)

# remove attributes (2) and cross-validate J48
atts = "RI|Na|Mg|Ca|Ba|Type"
flt = Filter(classname="weka.filters.unsupervised.attribute.RemoveByName", options=["-E", "(" + atts + ")", "-V"])
flt.inputformat(data)
filtered = flt.filter(data)
cls = Classifier(classname="weka.classifiers.trees.J48")
evl = Evaluation(filtered)
evl.crossvalidate_model(cls, filtered, 10, Random(1))
print(atts + ": %0.0f%%" % evl.percent_correct)

setups = (
    (
        ["-F", "10", "-T", "-1", "-B", "weka.classifiers.trees.J48"],
        ["-D", "0"]
    ),
    (
        ["-F", "10", "-T", "-1", "-B", "weka.classifiers.trees.J48"],
        ["-D", "0", "-N", "1"]
    ),
    (
        ["-F", "10", "-T", "-1", "-B", "weka.classifiers.trees.J48"],
        ["-D", "1", "-N", "2"]
    ),
    (
        ["-F", "10", "-T", "-1", "-B", "weka.classifiers.trees.J48"],
        ["-D", "2", "-N", "2"]
    ),
)

# attribute selection
for setup in setups:
    evl, search = setup
    aseval = ASEvaluation(classname="weka.attributeSelection.WrapperSubsetEval",
                          options=evl)
    assearch = ASSearch(classname="weka.attributeSelection.BestFirst",
                        options=search)
    print("\n--> Attribute selection\n")
    print(aseval.to_commandline())
    print(assearch.to_commandline())
    attsel = AttributeSelection()
    attsel.evaluator(aseval)
    attsel.search(assearch)
    attsel.select_attributes(data)
    print(attsel.results_string)

# cross-validation
aseval = ASEvaluation(classname="weka.attributeSelection.WrapperSubsetEval",
                      options=["-F", "10", "-B", "weka.classifiers.trees.J48"])
assearch = ASSearch(classname="weka.attributeSelection.BestFirst",
                    options=["-D", "0", "-N", "5"])
print("\n--> Attribute selection (cross-validation)\n")
print(aseval.to_commandline())
print(assearch.to_commandline())
attsel = AttributeSelection()
attsel.evaluator(aseval)
attsel.search(assearch)
attsel.crossvalidation(True)
attsel.select_attributes(data)
print(attsel.results_string)

jvm.stop()
