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

# More Data Mining with Weka - Class 5.3
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
import os
data_dir = os.environ.get("WEKAMOOC_DATA")
if data_dir is None:
    data_dir = "." + os.sep + "data"

import sys
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation, FilteredClassifier
from weka.filters import Filter
import weka.plot as plot
if plot.matplotlib_available:
    import matplotlib.pyplot as plt

jvm.start()

# load glass
fname = data_dir + os.sep + "glass.arff"
print("\nLoading dataset: " + fname + "\n")
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

# compute baseline
evl = Evaluation(data)
evl.crossvalidate_model(Classifier("weka.classifiers.rules.ZeroR"), data, 10, Random(1))
baseline = evl.percent_correct

# generate learning curves
percentages = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
repetitions = [1, 10, 100]
curves = {}
for repetition in repetitions:
    # progress info
    sys.stdout.write("Repetitions=" + str(repetition))
    # initialize curve
    curve = {}
    for percentage in percentages:
        curve[percentage] = 0
    curves[repetition] = curve
    # run and add up percentage correct from repetition
    for seed in xrange(repetition):
        seed += 1
        sys.stdout.write(".")
        for percentage in percentages:
            cls = Classifier(classname="weka.classifiers.trees.J48")
            flt = Filter(classname="weka.filters.unsupervised.instance.Resample",
                         options=["-Z", str(percentage), "-no-replacement"])
            fc = FilteredClassifier()
            fc.classifier = cls
            fc.filter = flt
            evl = Evaluation(data)
            evl.crossvalidate_model(fc, data, 10, Random(seed))
            curve[percentage] += (evl.percent_correct / repetition)
    # progress info
    sys.stdout.write("\n")

# output the results
if not plot.matplotlib_available:
    print("ZeroR: " + str(baseline))
    for repetition in repetitions:
        y = []
        for percentage in percentages:
            y.append(curves[repetition][percentage])
        print("Repetitions = " + str(repetition) + ":\n" + str(y))
else:
    fig, ax = plt.subplots()
    title = "Learning curve (J48 on glass)"
    fig.canvas.set_window_title(title)
    ax.set_title(title)
    ax.set_xlabel("training data %")
    ax.set_ylabel("accuracy %")
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 100])
    ax.plot([0, 100], [baseline, baseline], ls="--", c="0.3", label="ZeroR")
    ax.grid(True)
    for repetition in repetitions:
        y = []
        for percentage in percentages:
            y.append(curves[repetition][percentage])
        ax.plot(percentages, y, label="Rep=" + str(repetition))
        plt.draw()
    ax.legend()
    plt.show()

jvm.stop()
