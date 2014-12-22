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

# More Data Mining with Weka - Class 3.4
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.associations import Associator

jvm.start()

# load weather.nominal
fname = data_dir + os.sep + "weather.nominal.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# build Apriori (output as many rules as possible that min support of 0.01)
print("\n--> Output many weather rules\n")
assoc = Associator(classname="weka.associations.Apriori", options=["-M", "0.01", "-N", "10000"])
assoc.build_associations(data)
print(assoc)

# build Apriori (default options, outputting itemsets)
print("\n--> Output weather itemsets\n")
assoc = Associator(classname="weka.associations.Apriori", options=["-I"])
assoc.build_associations(data)
print(assoc)

# load supermarket
fname = data_dir + os.sep + "supermarket.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# build Apriori (default options, outputting itemsets)
print("\n--> Market basket analysis\n")
assoc = Associator(classname="weka.associations.Apriori", options=[])
assoc.build_associations(data)
print(assoc)

jvm.stop()
