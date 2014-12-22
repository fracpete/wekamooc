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

# More Data Mining with Weka - Class 1.3
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import os
import tempfile
import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.experiments import SimpleCrossValidationExperiment, SimpleRandomSplitExperiment, ResultMatrix, Tester
from weka.classifiers import Classifier

jvm.start()

# setup cross-validation experiment and run it
datasets = [
    data_dir + os.sep + "iris.arff"
]
classifiers = [
    Classifier(classname="weka.classifiers.trees.J48"),
    Classifier(classname="weka.classifiers.rules.OneR"),
    Classifier(classname="weka.classifiers.rules.ZeroR")
]
outfile = tempfile.gettempdir() + os.sep + "results-cv1.arff"
exp = SimpleCrossValidationExperiment(
    datasets=datasets,
    classifiers=classifiers,
    folds=10,
    runs=10,
    result=outfile
)
exp.setup()
exp.run()

# evaluate experiment
loader = converters.loader_for_file(outfile)
data = loader.load_file(outfile)
matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
comparison_col = data.attribute_by_name("Percent_correct").index
tester.instances = data
tester.resultmatrix = matrix
print(tester.header(comparison_col))
print(tester.multi_resultset_full(0, comparison_col))

# setup cross-validation experiment and run it
datasets = [
    data_dir + os.sep + "iris.arff",
    data_dir + os.sep + "breast-cancer.arff",
    data_dir + os.sep + "credit-g.arff",
    data_dir + os.sep + "diabetes.arff",
    data_dir + os.sep + "glass.arff",
    data_dir + os.sep + "ionosphere.arff",
    data_dir + os.sep + "segment-challenge.arff",
]
classifiers = [
    Classifier(classname="weka.classifiers.trees.J48"),
    Classifier(classname="weka.classifiers.rules.OneR"),
    Classifier(classname="weka.classifiers.rules.ZeroR")
]
outfile = tempfile.gettempdir() + os.sep + "results-cv2.arff"
exp = SimpleCrossValidationExperiment(
    datasets=datasets,
    classifiers=classifiers,
    folds=10,
    runs=10,
    result=outfile
)
exp.setup()
exp.run()

# evaluate experiment (1)
loader = converters.loader_for_file(outfile)
data = loader.load_file(outfile)
matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
comparison_col = data.attribute_by_name("Percent_correct").index
tester.instances = data
tester.resultmatrix = matrix
print(tester.header(comparison_col))
print(tester.multi_resultset_full(0, comparison_col))

# evaluate experiment (2) with datasets/classifiers swapped
loader = converters.loader_for_file(outfile)
data = loader.load_file(outfile)
matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
comparison_col = data.attribute_by_name("Percent_correct").index
tester.instances = data
tester.resultmatrix = matrix
tester.dataset_columns = ["Key_Scheme", "Key_Scheme_options", "Key_Scheme_version_ID"]
tester.result_columns = ["Key_Dataset"]
print(tester.header(comparison_col))
print(tester.multi_resultset_full(0, comparison_col))

jvm.stop()
