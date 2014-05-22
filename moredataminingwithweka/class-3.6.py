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

# More Data Mining with Weka - Class 3.6
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# TODO
# wherever your datasets are located
data_dir = "/some/where/data"

import os
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.clusterers import Clusterer, ClusterEvaluation
from weka.filters import Filter
import weka.plot.clusterers as plc

jvm.start()

# load iris
fname = data_dir + os.sep + "iris.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)

# remove class attribute
flt = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last"])
flt.set_inputformat(data)
filtered = flt.filter(data)

# build KMeans
print("\n--> SimpleKMeans\n")
cl = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
cl.build_clusterer(filtered)
evl = ClusterEvaluation()
evl.set_model(cl)
evl.test_model(filtered)
print(evl.get_cluster_results())
plc.plot_cluster_assignments(evl, data, atts=[], inst_no=True, wait=True)

# use AddCluster filter
print("\n--> AddCluster filter\n")
flt = Filter(classname="weka.filters.unsupervised.attribute.AddCluster",
             options=["-W", "weka.clusterers.SimpleKMeans -N 3"])
flt.set_inputformat(filtered)
addcl = flt.filter(filtered)
print(addcl)

# classes-to-clusters evaluation
print("\n--> Classes to clusters\n")
data.set_class_index(data.num_attributes() - 1)
cl = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
cl.build_clusterer(filtered)  # WITHOUT class attribute
evl = ClusterEvaluation()
evl.set_model(cl)
evl.test_model(data)  # WITH class attribute
print(evl.get_cluster_results())

jvm.stop()
