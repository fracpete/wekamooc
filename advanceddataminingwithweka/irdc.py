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

# Advanced Data Mining with Weka - IRDC shootout
# Copyright (C) 2015 Fracpete (fracpete at gmail dot com)

import os
import csv
import weka.core.jvm as jvm
from weka.core.dataset import Attribute, Instance, Instances
from weka.core.converters import Saver, Loader
from weka.classifiers import Classifier, FilteredClassifier, Evaluation
from weka.filters import Filter

data_dir = os.environ.get("IRDC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

jvm.start()

# convert csv to arff
print("Convert...")

# Files to use
# Cal(ibration) = Train
# Test = Test
# Val(idation) = to predict, missing reference values
files = ["DataSet1_Cal.csv", "DataSet1_Test.csv", "DataSet1_Val.csv",
         "DataSet2_Cal.csv", "DataSet2_Test.csv", "DataSet2_Val.csv"]

for infile in files:
    with open(data_dir + os.sep + infile, "rb") as csvfile:
        print(infile)
        outfile = os.path.splitext(infile)[0] + ".arff"
        reader = csv.reader(csvfile)
        data = None
        ref_present = True
        for index, row in enumerate(reader):
            if index == 0:
                atts = []
                ref_present = ("Reference value" in row) or ("Reference Value" in row)
                for idx, col in enumerate(row):
                    col = col.lower()
                    atts.append(Attribute.create_numeric(col))
                    if not ref_present and (idx == 0):
                        atts.append(Attribute.create_numeric("reference value"))
                data = Instances.create_instances("irdc", atts, 0)
            else:
                values = []
                for idx, col in enumerate(row):
                    values.append(float(col))
                    if not ref_present and (idx == 0):
                        values.append(float('NaN'))
                inst = Instance.create_instance(values)
                data.add_instance(inst)

        saver = Saver(classname="weka.core.converters.ArffSaver")
        saver.save_file(data, data_dir + os.sep + outfile)

# train/test/predict
print("Train/test/predict...")

groups = ["DataSet1", "DataSet2"]
# groups = ["DataSet2"]

for group in groups:
    print(group)
    train = data_dir + os.sep + group + "_Cal.arff"
    test = data_dir + os.sep + group + "_Test.arff"
    pred = data_dir + os.sep + group + "_Val.arff"

    loader = Loader(classname="weka.core.converters.ArffLoader")
    print(train)
    train_data = loader.load_file(train)
    train_data.class_index = train_data.attribute_by_name("reference value").index
    print(test)
    test_data = loader.load_file(test)
    test_data.class_index = test_data.attribute_by_name("reference value").index
    print(pred)
    pred_data = loader.load_file(pred)
    pred_data.class_index = pred_data.attribute_by_name("reference value").index

    cls = FilteredClassifier()
    cls.classifier = Classifier(classname="weka.classifiers.functions.LinearRegression", options=["-S", "1", "-C"])
    cls.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    cls.build_classifier(train_data)
    evl = Evaluation(train_data)
    evl.test_model(cls, test_data)
    print(evl.summary())

jvm.stop()