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

# More Data Mining with Weka - Class 5.5
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import tempfile
import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver

jvm.start()

# load iris (gets rid of all the comments)
fname = data_dir + os.sep + "iris.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# output arff
outfile = tempfile.gettempdir() + os.sep + "test.arff"
saver = Saver(classname="weka.core.converters.ArffSaver")
saver.save_file(data, outfile)
f = open(outfile, 'r')
arff = f.read()
f.close()
print(arff)

# output xrff
outfile = tempfile.gettempdir() + os.sep + "test.xrff"
saver = Saver(classname="weka.core.converters.XRFFSaver")
saver.save_file(data, outfile)
f = open(outfile, 'r')
xrff = f.read()
f.close()
print(xrff)

jvm.stop()
