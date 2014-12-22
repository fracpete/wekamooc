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

# Data Mining with Weka - Class 1.5
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
from weka.core.dataset import Instances
from weka.filters import Filter

jvm.start()

# load weather.nominal
fname = data_dir + os.sep + "weather.nominal.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)

# output header
print(Instances.template_instances(data))

# remove attribute no 3
print("\nRemove attribute no 3")
fltr = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "3"])
fltr.inputformat(data)
filtered = fltr.filter(data)

# output header
print(Instances.template_instances(filtered))

# save modified dataset
saver = Saver(classname="weka.core.converters.ArffSaver")
saver.save_file(filtered, data_dir + os.sep + "weather.nominal-filtered.arff")

jvm.stop()

