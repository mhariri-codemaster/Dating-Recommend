"""
Collaborative Filtering ALS Recommender System using Spark MLlib.

Usage:
    ./recommend.py train <ratings_file> <model_save_file> [--partitions=<n>]
                   [--ranks=<n>] [--lambdas=<n>] [--iterations=<n>]
    ./recommend.py recommend <ratings_file> <gender_file> <model_save_file>
                   <userID> [--partitions=<n>]
    ./recommend.py (-h | --help)

Options:
    -h, --help         Show this screen and exit.
    --partitions=<n>   Partition count [Default: 4]
    --ranks=<n>        List of ranks [Default: 6,8,12]
    --lambdas=<n>      List of lambdas [Default: 0.1,1.0,10.0]
    --iterations=<n>   List of iterations [Default: 10,20]

Examples:
    bin/spark-submit recommend.py train ratings.dat model_path
    bin/spark-submit recommend.py recommend ratings.dat gender.dat model_path userID

Credits:
    original code cloned from:
        https://github.com/marklit/recommend
"""

import contextlib
import itertools
from math import sqrt
from operator import add
import sys

from docopt import docopt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel


SPARK_EXECUTOR_MEMORY = '2g'
SPARK_APP_NAME = 'dateRecommender'
SPARK_MASTER = 'local'


@contextlib.contextmanager
def spark_manager():
    conf = SparkConf().setMaster(SPARK_MASTER) \
                      .setAppName(SPARK_APP_NAME) \
                      .set("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
    spark_context = SparkContext(conf=conf)

    try:
        yield spark_context
    finally:
        spark_context.stop()

def compute_rmse(model, data, validation_count):
    """
    Compute RMSE (Root Mean Squared Error).

    :param object model:
    :param RDD data:
    :param integer validation_count:
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = \
        predictions.map(lambda x: ((x[0], x[1]), x[2])) \
                   .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
                   .values()
    return sqrt(
        predictionsAndRatings.map(
            lambda x: (x[0] - x[1]) ** 2
        ).reduce(add) / float(validation_count)
    )

def train(ratings_path, model_save_path, numPartitions, ranks, lambdas, iterations):
    """
    Train the model using ALS using the provided ratings
    
    :param str ratings_path: file location of ratings.dat
    :param str model_save_path: full path where to save final model
    :param int numPartitions: number of partitions to use
    :param list ranks: list of ranks to use
    :param list lambdas: list of lambdas to use
    :param list iterations: list of iteration counts
    """

    with spark_manager() as sc:
	print ("Starting Spark Manager")

	print ("Loading ratings dataset")
	ratings = sc.textFile(ratings_path) \
            	    .map(lambda line: line.split(",")) \
                    .filter(lambda x: len(x)==3) \
                    .map(lambda x: Rating(int(x[0]), int(x[1]), int(x[2])))

        numRatings = ratings.count()

        numUsers = ratings.map(lambda r: r[0]) \
                          .distinct() \
                          .count()

        numProfiles = ratings.map(lambda r: r[1]) \
                             .distinct() \
                             .count()
	
	# Use this trick below to sample the dataset into 3 sets
	print("Preparing training, validation and test datasets")
        training = ratings.filter(lambda x: (x[0]+x[1])%10 < 6) \
                          .repartition(numPartitions) \
                          .cache()

        validation = ratings.filter(lambda x: (x[0]+x[1])%10 >= 6 and (x[0]+x[1])%10 < 8) \
                            .repartition(numPartitions) \
                            .cache()

        test = ratings.filter(lambda x: (x[0]+x[1])%10 >= 8) \
                      .cache()

        numTraining = training.count()
        numValidation = validation.count()
        numTest = test.count()

	# Find the best model
	print("Iterating through different models")
        bestValidationRmse = float("inf")
        bestModel, bestRank, bestLambda, bestNumIter = None, 0, -1.0, -1

        for rank, lmbda, numIter in itertools.product(ranks,
                                                      lambdas,
                                                      iterations):
            model = ALS.train(ratings=training,
                              rank=rank,
                              iterations=numIter,
                              lambda_=lmbda)

            validationRmse = compute_rmse(model, validation, numValidation)

            if validationRmse < bestValidationRmse:
                bestModel, bestValidationRmse = model, validationRmse
                bestRank, bestLambda, bestNumIter = rank, lmbda, numIter
	print("Best model found")

        # Evaluate the best model on the test set
        testRmse = compute_rmse(bestModel, test, numTest)
	
        # save the model trained on full set
	training = ratings.repartition(numPartitions) \
                          .cache()
        save_model = ALS.train(ratings=training,
                               rank=bestRank,
                               iterations=bestNumIter,
                               lambda_=bestLambda)
	save_model.save(sc, model_save_path)

    print
    print 'Ratings:     {:10,}'.format(numRatings)
    print 'Users:       {:10,}'.format(numUsers)
    print 'Movies:      {:10,}'.format(numProfiles)
    print
    print 'Training:    {:10,}'.format(numTraining)
    print 'Validation:  {:10,}'.format(numValidation)
    print 'Test:        {:10,}'.format(numTest)
    print
    print 'The best model was trained with:'
    print '    Rank:             {:10,}'.format(bestRank)
    print '    Lambda:           {:10,.6f}'.format(bestLambda)
    print '    Iterations:       {:10,}'.format(bestNumIter)
    print '    RMSE on test set: {:10,.6f}'.format(testRmse)


def recommend(ratings_path, gender_path, model_path,
	      userID, numPartitions):
    """
    Recommend dating profiles for user based on previous ratings
    
    :param str ratings_path: file location of ratings.dat
    :param str gender_path: file location of gender.dat
    :param str model_path: file location of saved model from training

    :param int userID: ID of user to recommend profiles for
    :param int numPartitions: number of partitions to use
    """

    with spark_manager() as sc:
	print ("Starting Spark Manager")

	print ("Loading saved model")
        model = MatrixFactorizationModel.load(sc, model_path)

	print ("Collect ratings of user "+str(userID))
	rated = sc.textFile(ratings_path) \
            	  .map(lambda line: line.split(",")) \
                  .filter(lambda x: len(x)==3) \
                  .map(lambda x: (int(x[0]), int(x[1]))) \
		  .filter(lambda x: x[0]==userID) \
                  .map(lambda x: x[1]) \
                  .collect()

	print("Get complete list of profiles")
        profiles = sc.textFile(ratings_path) \
            	     .map(lambda line: line.split(",")) \
                     .filter(lambda x: len(x)==3) \
		     .map(lambda x: int(x[1])) \
                     .distinct() \
                     .filter(lambda x: x not in rated) \
		     .map(lambda x: (userID, x)) \
                     .repartition(numPartitions) \
                     .cache()

	print("Performing predictions for user "+str(userID))
        predictions = model.predictAll(profiles).collect()

        # Get the recommendations
        recommendations = sorted(predictions,
                                 key=lambda x: x[2],
                                 reverse=True)

	# Get the genders of the profiles
	print("Loading gender dataset")
	genders = sc.textFile(gender_path) \
            	  .map(lambda line: line.split(",")) \
                  .filter(lambda x: len(x)==2) \
                  .map(lambda x: (int(x[0]), str(x[1])))
	
        # Make the recommendations matching Male to Female when available
    	print ("Recommended profiles for user "+str(userID))
    	user_gender = genders.lookup(userID)
    	if len(user_gender) == 0 or user_gender[0]=='U': # gender not listed
            for rec in recommendations[:10]:
            	print ("profile: "+str(rec[1])+", expected rating: "+str(rec[2]))
    	elif user_gender[0] == 'M':
	    count = 0
            for rec in recommendations:
	    	profile_gender = genders.lookup(rec[1])
	    	if len(profile_gender) >= 0 and profile_gender[0]=='F':
                    print ("profile: "+str(rec[1])+", expected rating: "+str(rec[2]))
		    count+=1
		if count>10:
                    break
    	else:
	    count = 0
            for rec in recommendations:
	    	profile_gender = genders.lookup(rec[1])
	    	if len(profile_gender) >= 0 and profile_gender[0]=='M':
                    print ("profile: "+str(rec[1])+", expected rating: "+str(rec[2]))
		    count+=1
		if count>10:
                    break

def main(argv):
    """
    :param dict argv: command line arguments
    """
    opt = docopt(__doc__, argv)
	
    if "--partitions" in opt:
	try:
           int(opt["--partitions"])
	except ValueError:
           print("The '--partitions' option allows for one integer value only")
           return

    if opt['train']:
        ranks    = [int(rank)      for rank in opt['--ranks'].split(',')]
        lambdas  = [float(_lambda) for _lambda in opt['--lambdas'].split(',')]
        iterations = [int(_iter)   for _iter in opt['--iterations'].split(',')]

        train(opt['<ratings_file>'],
              opt['<model_save_file>'],
              int(opt['--partitions']),
              ranks,
              lambdas,
              iterations)

    if opt['recommend']:
        recommend(opt['<ratings_file>'],
                  opt['<gender_file>'],
                  opt['<model_save_file>'],
                  int(opt['<userID>']),
                  int(opt['--partitions']))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        pass
