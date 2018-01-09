# Introduction

This is my attempt at using Spark ALS to make some recommendations for users of a dating website. The algorithm uses the ratings that users made over certain profiles as well as those made by others over other profiles to see which profiles are the most probable matches for those users. The algorithm selects from the pool of profiles not previously rated by the user and tries to match males to females when that information is available. 

# Requirements

I ran this using:  
On Ubuntu 16.04  
docopt 0.6.2  
spark 2.2.1  
pyspark  
python 2.7.12  

## Dating profiles ratings data

Two sets of data are available. The ratings.dat has the ratings of the users to the profiles with schema (user, profile, rating). The gender.dat has the gender information of the users and profiles with schema (ID, gender). Both are avilable at:  
  
http://www.occamslab.com/petricek/data/

# Example Uses

## Training

```bash
$ python recommend.py train ratings.dat model_save_path
```

```bash
$ python recommend.py train ratings.dat model_save_path \
    --ranks=8,10,12 --lambdas=0.01,0.1,1.0 --iterations=10,15 --partitions=6
```

## Recommending

```bash
$ python recommend.py recommend ratings.dat gender.dat model_save_path 100
```

```bash
$ python recommend.py recommend ratings.dat gender.dat model_save_path \
   100 --partitions=3
```
