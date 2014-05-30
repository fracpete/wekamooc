/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Data Mining with Weka - Class 2.4
 * Copyright (C) 2014 Fracpete (fracpete at gmail dot com)
 */

# Use the WEKAMOOC_DATA environment variable to set the location 
# for the datasets
data_dir = System.getenv("WEKAMOOC_DATA")
if (data_dir == null)
  data_dir = "./data"

import java.io.File
import java.util.Random
import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.core.converters.ConverterUtils.DataSource
import weka.core.Instances

classifiers = [
    new weka.classifiers.rules.ZeroR(),
    new weka.classifiers.trees.J48(),
    new weka.classifiers.bayes.NaiveBayes(),
    new weka.classifiers.lazy.IBk(),
    new weka.classifiers.rules.PART()
]
percent = 66.0

// load diabetes
fname = data_dir + File.separator + "diabetes.arff"
printf("\nLoading dataset: " + fname + "\n")
data = DataSource.read(fname)
data.setClassIndex(data.numAttributes() - 1)

// random split on data
rnd = new Random(1)
data.randomize(rnd)
train_size = (int) (data.numInstances() * percent / 100)
test_size = data.numInstances() - train_size
train = new Instances(data, 0, train_size)
test = new Instances(data, train_size, test_size)

// train + evaluate
for (classifier in classifiers) {
  evl = new Evaluation(data)
  classifier.buildClassifier(train)
  evl.evaluateModel(classifier, test)
  printf(classifier.getClass().getName() + ": %02.0f\n", evl.pctCorrect())
}

// load supermarket
fname = data_dir + File.separator + "supermarket.arff"
printf("\nLoading dataset: " + fname + "\n")
data = DataSource.read(fname)
data.setClassIndex(data.numAttributes() - 1)

// random split on data
rnd = new Random(1)
data.randomize(rnd)
train_size = (int) (data.numInstances() * percent / 100)
test_size = data.numInstances() - train_size
train = new Instances(data, 0, train_size)
test = new Instances(data, train_size, test_size)

// train + evaluate
for (classifier in classifiers) {
  evl = new Evaluation(data)
  classifier.buildClassifier(train)
  evl.evaluateModel(classifier, test)
  printf(classifier.getClass().getName() + ": %02.0f\n", evl.pctCorrect())
}
