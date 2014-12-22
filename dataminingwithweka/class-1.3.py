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

# Data Mining with Weka - Class 1.3
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import weka.core.jvm as jvm
from weka.core.converters import Loader

jvm.start()

# load weather.nominal
fname = data_dir + os.sep + "weather.nominal.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)

# print data
print(data)

# print attributes
print("\nAttributes")
for i in xrange(data.num_attributes):
    att   = data.attribute(i)
    stats = data.attribute_stats(i)
    extra = ""
    if att.is_nominal:
        for n in xrange(att.num_values):
            if n > 0:
                extra += ", "
            extra += "%s=%d" % (att.value(n), stats.nominal_counts[n])
    elif att.is_numeric:
        nstats = stats.numeric_stats()
        extra = "min=%0.4f, max=%0.4f, mean=%0.4f, stddev=%0.4f" % \
                (nstats.min(), nstats.max(), nstats.mean(), nstats.stddev())
    print(str(i+1) + ". " + att.name + ": " + att.type_str(True) + " [" + str(extra) + "]")

# load glass
fname = data_dir + os.sep + "glass.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)

# print data
print(data)

# print attributes
print("\nAttributes")
for i in xrange(data.num_attributes):
    att   = data.attribute(i)
    stats = data.attribute_stats(i)
    extra = ""
    if att.is_nominal:
        for n in xrange(att.num_values):
            if n > 0:
                extra += ", "
            extra += "%s=%d" % (att.value(n), stats.nominal_counts[n])
    elif att.is_numeric:
        nstats = stats.numeric_stats
        extra = "min=%0.4f, max=%0.4f, mean=%0.4f, stddev=%0.4f" % \
                (nstats.min, nstats.max, nstats.mean, nstats.stddev)
    print(str(i+1) + ". " + att.name + ": " + att.type_str(True) + " [" + str(extra) + "]")

jvm.stop()
