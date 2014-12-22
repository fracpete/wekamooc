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

# Data Mining with Weka - Class 5.1
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.filters import Filter

jvm.start()

# load iris
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "iris.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.class_is_last()

# AddExpression
print("Computing area fromm petallength and petalwidth")
fltr = Filter(classname="weka.filters.unsupervised.attribute.AddExpression", options=["-E", "a3*a4", "-N", "petalarea"])
fltr.inputformat(data)
filtered = fltr.filter(data)
print(filtered)

# Normalize
print("Normalize numeric attributes")
fltr = Filter(classname="weka.filters.unsupervised.attribute.Normalize")
fltr.inputformat(data)
filtered = fltr.filter(data)
print("sepallength stats (original)")
print(data.attribute_stats(0).numeric_stats)
print("sepallength stats (normalized)")
print(filtered.attribute_stats(0).numeric_stats)

# Standardize
print("Standardize numeric attributes")
fltr = Filter(classname="weka.filters.unsupervised.attribute.Standardize")
fltr.inputformat(data)
filtered = fltr.filter(data)
print("sepallength stats (original)")
print(data.attribute_stats(0).numeric_stats)
print("sepallength stats (normalized)")
print(filtered.attribute_stats(0).numeric_stats)

# Discretize
print("Discretize numeric attributes (supervised)")
fltr = Filter(classname="weka.filters.supervised.attribute.Discretize")
fltr.inputformat(data)
filtered = fltr.filter(data)
print(filtered)

# PCA
print("Principal components analysis")
fltr = Filter(classname="weka.filters.unsupervised.attribute.PrincipalComponents")
fltr.inputformat(data)
filtered = fltr.filter(data)
print(filtered)

# load anneal
loader = Loader(classname="weka.core.converters.ArffLoader")
fname = data_dir + os.sep + "anneal.arff"
print("\nLoading dataset: " + fname + "\n")
data = loader.load_file(fname)
data.class_is_last()

# RemoveUseless
print("RemoveUseless")
fltr = Filter(classname="weka.filters.unsupervised.attribute.RemoveUseless")
fltr.inputformat(data)
filtered = fltr.filter(data)
print("Original header (#att=" + str(data.num_attributes) + "):\n" + str(Instances.template_instances(data)))
print("Filtered header (#att=" + str(filtered.num_attributes) + "):\n" + str(Instances.template_instances(filtered)))

jvm.stop()
