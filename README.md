A Distributed Evolutionary Multivariate Discretizer (DEMD)
==========================================================

Here, a Distributed Evolutionary Multivariate Discretizer (DEMD) for data reduction on Spark is presented. In order to alleviate the complexity derived from the evolutionary process, the whole evalution phase in this algorithm has been fully parallelized. For this purpose, both the set of chromosomes and instances have been split into different partitions and a random cross-evaluation process between them has been performed. 

Despite the non-deterministic nature of the algorithm, the evalution results has shown to yield good discretization schemes. A thorough experimental evaluation performed using several huge datasets (up to O(10^7) instances and O(10^4) features) has shown the usefulness of our approach.

Spark package: http://spark-packages.org/package/sramirez/spark-DEMD-discretizer

## Parameters:

Our distributed approach includes several user-defined input parameters, which are described below:
* train: the raw dataset, in RDD format.
* contFeat: sequence of feature indexes to discretize.
* mvfactor: numerical ratio between the number of feature chunks and the
number of data partitions. It is a integer value, which can be greater
than 1 (default).
* alpha: a weight factor for the fitness function used in the inner evolution-
ary process (range [0, 1]).
* nChr: Number of chromosome evaluations to be performed in each process
(evolutionary algorithm).
* srate: Percentage of sampled instances used to evaluate the candidate
points (range [0, 1]).
* vth: Percentage of points selected in each aggregation process (range
[0, 1]).


## Example: 
	import org.apache.spark.mllib.feature._
  
	val train = /** A RDD of LabeledPoint **/
  	val contFeat = (0 until train.first.features.size) /** Sequence of feature indexes to discretize **/
  	val nChr = 50
  	val ngeval = 5000
  	val mvfactor = 1
  	val alpha = .7f
  	val srate = .1f
  	val vth = 100
              
  	val discretizer = DEMDdiscretizer.train(train,
        	contFeat,
        	nChr,
        	ngeval,
        	alpha,
        	mvfactor,
        	srate,
        	vth) 
    	
	val discretized = data.map(i => LabeledPoint(i.label, discretizer.transform(i.features)))
	discretized.first()


## Contributors

- Sergio Ramírez-Gallego (sramirez@decsai.ugr.es) (main contributor and maintainer).

