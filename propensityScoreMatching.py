from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from math import exp
from itertools import chain
from scipy.stats import ttest_ind
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import matplotlib
import matplotlib.pyplot as plt


#current feature set
data_all = spark.sql("SELECT test_control, feature_1, feature_2, feature_N \
from scheme.table \
where criteria = met")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="test_control", outputCol="indexedLabel").fit(data_all)

ignore = ['features_iDontWant']
assembler = VectorAssembler(
    inputCols=[x for x in data_all.columns if x not in ignore],
    outputCol='features')

data_new = assembler.transform(data_all)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data_new)

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=200, featureSubsetStrategy = '5')

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf]) # Chain indexers and forest in a Pipeline

model = pipeline.fit(data_new) # Train model.  This also runs the indexers.

gsts_final_df = model.transform(data_new) # Make predictions.

test_all = gsts_final_df[(gsts_final_df['test_control'] == 1)] #test guests
control_all = gsts_final_df[(gsts_final_df['test_control'] == 0)] #control population

test_all_sort = test_all.sort('probability', ascending=1) #sort by propensity
control_all_sort = control_all.sort('probability', ascending=1) #sort by propensity

t_prop_df = pd.DataFrame(test_all.rdd.map(lambda x: x[39]).map(lambda x: x[1]).take(100000))
NN_prop_df = pd.DataFrame(control_all.rdd.map(lambda x: x[39]).map(lambda x: x[1]).take(100000))

#plot propensity
t_prop_df.plot(kind="density", alpha=0.7, color='k', label='Test')
NN_prop_df.plot(kind="density", ax=plt.gca(),alpha=0.4,color='r', label='Control')

plt.legend(['Test', 'Control'])
plt.title('Propensity Score')
plt.xlim((0, 1))
plt.show()

#zip with index for filtering
test_with_index = test_all_sort.rdd.zipWithIndex()
test_count = test_all_sort.count()

part1 = test_count / 3 #partition 1
part2 = 2 * test_count / 3 #partition 2

#split file into thirds
#these smaller groupings will be used as neighbors k-NN algorithm
#limiting dataset so we don't crash the driver

#partition 1
test1 = test_with_index.filter(lambda x: x[1] <  \
                               part1).map(lambda x: x[0]).collect()

#partition 2
test2 = test_with_index.filter(lambda x: x[1] <  \
                               part2).filter(lambda x: x[1] >= \
                               part1).map(lambda x: x[0]).collect()

#partition 3
test3 = test_with_index.filter(lambda x: x[1] >= part2).map(lambda x: x[0]).collect()

#converting RDD to a pandas DF for sklearn k-NN algo implementation
test1df = sc.parallelize(test1).toDF()
test1_all = test1df.toPandas()
test1_f = test1df.rdd.map(lambda x: x[39]).map(lambda x: x[1]).collect()

test2df = sc.parallelize(test2).toDF()
test2_all = test2df.toPandas()
test2_f = test2df.rdd.map(lambda x: x[39]).map(lambda x: x[1]).collect()

test3df = sc.parallelize(test3).toDF()
test3_all = test3df.toPandas()
test3_f = test3df.rdd.map(lambda x: x[39]).map(lambda x: x[1]).collect()

#using only a sample of the total control population for matching
#plenty of controls to chose from, we don't need them all

weights = [1.0, 7.0]
control_sample, control_rest = control_all_sort.randomSplit(weights)

#converting to pandas for sklearn
control1_all = control_sample.toPandas()

#collect only the propensity score
control1_probs = control_sample.rdd.map(lambda x: x[39]).map(lambda x: x[1])
control1_pDF = pd.DataFrame(control1_probs.collect())

#fitting k-NN on control
nbrs1 = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control1_pDF)

#broadcast the fit to all the mappers
bc_nbrs1 = sc.broadcast(nbrs1)

test_1 = sc.parallelize(test1_f)
test_2 = sc.parallelize(test2_f)
test_3 = sc.parallelize(test3_f)

#map the test guests with the fitted control
#return euclidean distance for matching comparison
nn1 = test_1.map(lambda x: bc_nbrs1.value.kneighbors(x, return_distance=True))
nn2 = test_2.map(lambda x: bc_nbrs1.value.kneighbors(x, return_distance=True))
nn3 = test_3.map(lambda x: bc_nbrs1.value.kneighbors(x, return_distance=True))

#index location of neighbors
neighbors1 = nn1.map(lambda x: x[1])
neighbors2 = nn2.map(lambda x: x[1])
neighbors3 = nn3.map(lambda x: x[1])

#flattening index array into single index points
flatNeighbors1 = neighbors1.flatMap(lambda xs: chain(*xs))
flatNeighbors2 = neighbors2.flatMap(lambda xs: chain(*xs))
flatNeighbors3 = neighbors3.flatMap(lambda xs: chain(*xs))

#collecting control observations at index locations
c_neighbors1 = control1_all.iloc[flatNeighbors1.collect()]
c_neighbors2 = control1_all.iloc[flatNeighbors2.collect()]
c_neighbors3 = control1_all.iloc[flatNeighbors3.collect()]

#collecting probabilities at index location for control comparison
control_probs_1 = control1_pDF.iloc[flatNeighbors1.collect()]
control_probs_2 = control1_pDF.iloc[flatNeighbors2.collect()]
control_probs_3 = control1_pDF.iloc[flatNeighbors3.collect()]
