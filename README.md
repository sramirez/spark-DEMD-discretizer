A Distributed Evolutionary Multivariate Discretizer (DEMD)
==========================================================

The present algorithm is a distributed version of algorithm presented in [1], which is a evolutionary multivariate discretizer (EMD) for data reduction. This new distributed version has entailed a complete redesign of the original approach. In order to alleviate the complexity derived from the original algorithm, the whole evaluation process in this algorithm have 2been fully parallelized. For this purpose, both the set of chromosomes and instances have been split into different partitions and a random cross-evaluation process between them has been performed. 

Despite being different from the sequential approach, this type of evaluation has shown to yield good discretization schemes. To demonstrate the usefulness of this solution, a thorough experimental evaluation has been performed using several huge datasets (up to O(10^7) instances and O(10^4) features).

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

##References

[1] Ramirez-Gallego, S.; Garcia, S.; Benitez, J.M.; Herrera, F., "Multivariate Discretization Based on Evolutionary Cut Points Selection for Classification," in Cybernetics, IEEE Transactions on , vol.PP, no.99, pp.1-1 
doi: 10.1109/TCYB.2015.2410143
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7063251&isnumber=6352949
