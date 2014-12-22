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

# More Data Mining with Weka - Class 5.2
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import tempfile
import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation
from weka.experiments import SimpleCrossValidationExperiment, ResultMatrix, Tester

jvm.start()

# load weather.numeric
fname = data_dir + os.sep + "weather.numeric.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# cross-validate classifiers
classifiers = [
    "weka.classifiers.functions.MultilayerPerceptron",
    "weka.classifiers.trees.J48",
    "weka.classifiers.bayes.NaiveBayes",
    "weka.classifiers.functions.SMO",
    "weka.classifiers.lazy.IBk"
]
for classifier in classifiers:
    cls = Classifier(classname=classifier)
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, 10, Random(1))
    print("%s: %0.0f%%" % (classifier, evl.percent_correct))

# configure experiment
print("This will take some time, so grab a cuppa... And a muffin... And read the paper...")
datasets = [
    data_dir + os.sep + "iris.arff",
    data_dir + os.sep + "breast-cancer.arff",
    data_dir + os.sep + "credit-g.arff",
    data_dir + os.sep + "diabetes.arff",
    data_dir + os.sep + "glass.arff",
    data_dir + os.sep + "ionosphere.arff"
]
classifiers = [
    Classifier(classname="weka.classifiers.functions.MultilayerPerceptron"),
    Classifier(classname="weka.classifiers.rules.ZeroR"),
    Classifier(classname="weka.classifiers.rules.OneR"),
    Classifier(classname="weka.classifiers.trees.J48"),
    Classifier(classname="weka.classifiers.bayes.NaiveBayes"),
    Classifier(classname="weka.classifiers.lazy.IBk"),
    Classifier(classname="weka.classifiers.functions.SMO"),
    Classifier(classname="weka.classifiers.meta.AdaBoostM1"),
    # handles only 2-class problems: Classifier(classname="weka.classifiers.functions.VotedPerceptron")
]
outfile = tempfile.gettempdir() + os.sep + "results-cv.arff"   # store results for later analysis
exp = SimpleCrossValidationExperiment(
    classification=True,
    runs=10,
    folds=10,
    datasets=datasets,
    classifiers=classifiers,
    result=outfile)
exp.setup()
exp.run()
# evaluate previous run
loader = converters.loader_for_file(outfile)
data = loader.load_file(outfile)
matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
tester.resultmatrix = matrix
comparison_col = data.attribute_by_name("Percent_correct").index
tester.instances = data
print(tester.header(comparison_col))
print(tester.multi_resultset_full(0, comparison_col))

jvm.stop()
