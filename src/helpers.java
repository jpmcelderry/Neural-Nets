import java.util.*;

public class helpers {
	/*
	 * Method to check whether an array contains a given integer, returns the index where that item is found
	 */
	public static int arrayContains(int[] array, int query) {
		for (int index=0;index<array.length;index++) {
			if (array[index]==query) {
				return index;
			}
		}
		return -1;
	}
	
	public static String[] randomImpute(ArrayList<String[]> data, String[] toImpute, int classIndex, int attributeIndex) {
		List<String> list = new ArrayList<>();
		//iterate over the whole array to find samples with matching class as the data point to impute
		for (String[] sample: data) {
			//If the samples match and the other sample has a value, add it to the list
			if(sample[classIndex].equals(toImpute[classIndex]) && !sample[attributeIndex].equals("?")) {
				list.add(sample[attributeIndex]);
			}
		}
		//Only randomly select a value if the list isn't empty
		if(list.size()>0) {
			Random rand = new Random();
			String randomVal = list.get(rand.nextInt(list.size()));
			toImpute[attributeIndex] = randomVal;
		}
		//If no values were extracted, return the item as is
		else {
			return toImpute;
		}
		return toImpute;
	}
	
	public static ArrayList<ArrayList<classificationSample>> zScoreTransformC(ArrayList<ArrayList<classificationSample>> samples, int[][] categoricalFeatures) {
		double[] means = new double[samples.get(0).get(0).features.length];
		double[] standardDev = new double[samples.get(0).get(0).features.length];
		double sampleCount = 0;
		for(ArrayList<classificationSample> sampleSet: samples) {
			sampleCount += sampleSet.size();
		}
		//Calculate the mean of each feature
		for(ArrayList<classificationSample> set: samples) {
			for(classificationSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(arrayContains(categoricalFeatures[0],feature)==-1) {
						means[feature] += sample.features[feature]/sampleCount;
					}
				}
			}
		}
		//Finds variance from sum of squared difference
		for(ArrayList<classificationSample> set: samples) {
			for(classificationSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(arrayContains(categoricalFeatures[0],feature)==-1) {
						double squareDif = (sample.features[feature]-means[feature])*(sample.features[feature]-means[feature]);
						standardDev[feature] += squareDif/sampleCount;
					}
				}
			}
		}
		//Find standard deviation from variance
		for(int feature=0;feature<means.length;feature++) {
			if(arrayContains(categoricalFeatures[0],feature)==-1) {
				standardDev[feature] = Math.sqrt(standardDev[feature]);
			}
		}
		//Z-score normalization
		for(ArrayList<classificationSample> set: samples) {
			for(classificationSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(arrayContains(categoricalFeatures[0],feature)==-1) {
						sample.features[feature] = (double) (sample.features[feature]-means[feature])/standardDev[feature];
					}
				}
			}
		}
		
		return samples;
	}
	
	public static ArrayList<ArrayList<regressionSample>> zScoreTransformR(ArrayList<ArrayList<regressionSample>> samples, int[][] categoricalFeatures) {
		double[] means = new double[samples.get(0).get(0).features.length];
		double[] standardDev = new double[samples.get(0).get(0).features.length];
		double sampleCount = 0;
		for(ArrayList<regressionSample> sampleSet: samples) {
			sampleCount += sampleSet.size();
		}
		//Calculate the mean of each feature
		for(ArrayList<regressionSample> set: samples) {
			for(regressionSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(arrayContains(categoricalFeatures[0],feature)==-1) {
						means[feature] += sample.features[feature]/sampleCount;
					}
				}
			}
		}
		//Finds variance from sum of squared difference
		for(ArrayList<regressionSample> set: samples) {
			for(regressionSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(arrayContains(categoricalFeatures[0],feature)==-1) {
						double squareDif = (sample.features[feature]-means[feature])*(sample.features[feature]-means[feature]);
						standardDev[feature] += squareDif/sampleCount;
					}
				}
			}
		}
		//Find standard deviation from variance
		for(int feature=0;feature<means.length;feature++) {
			if(arrayContains(categoricalFeatures[0],feature)==-1) {
				standardDev[feature] = Math.sqrt(standardDev[feature]);
			}
		}
		//Z-score normalization
		for(ArrayList<regressionSample> set: samples) {
			for(regressionSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(arrayContains(categoricalFeatures[0],feature)==-1) {
						sample.features[feature] = (double) (sample.features[feature]-means[feature])/standardDev[feature];
					}
				}
			}
		}
		
		return samples;
	}
	
	/*
	 * Method to one-hot-encode using the information contained in 'categoricalFeatures' array
	 * real work is done by method 'oneHotEncode()'
	 * returns a 'classificationSample' arraylist
	 */
	public static ArrayList<ArrayList<classificationSample>> oneHotEncodeC(ArrayList<ArrayList<classificationSample>> samples, int[][] categoricalFeatures) {
		//Create arraylist to return
		ArrayList<ArrayList<classificationSample>> hotEncoded = new ArrayList<ArrayList<classificationSample>>();
		//Predict number of extra indices needed for one hot encoding
		int extraSpots=0;
		for(int feature=0;feature<categoricalFeatures[1].length;feature++) {
			extraSpots += categoricalFeatures[1][feature]-1;
		}
		
		//iterate by fold
		for(ArrayList<classificationSample> set: samples) {
			ArrayList<classificationSample> temp = new ArrayList<classificationSample>();
			
			//iterate by sample
			for(classificationSample sample: set) {
				temp.add(new classificationSample(sample.classification,oneHotEncode(sample.features,extraSpots,categoricalFeatures)));
			}
			hotEncoded.add(temp);
		}
		return hotEncoded;
	}
	
	/*
	 * Method to one-hot-encode using the information contained in 'categoricalFeatures' array
	 * real work is done by method 'oneHotEncode()'
	 * returns a 'regressionSample' arraylist
	 */
	public static ArrayList<ArrayList<regressionSample>> oneHotEncodeR(ArrayList<ArrayList<regressionSample>> samples, int[][] categoricalFeatures) {
		//Create arraylist to return
		ArrayList<ArrayList<regressionSample>> hotEncoded = new ArrayList<ArrayList<regressionSample>>();
		//Predict number of extra indices needed for one hot encoding
		int extraSpots=0;
		for(int feature=0;feature<categoricalFeatures[1].length;feature++) {
			extraSpots += categoricalFeatures[1][feature]-1;
		}
		
		//iterate by fold
		for(ArrayList<regressionSample> set: samples) {
			ArrayList<regressionSample> temp = new ArrayList<regressionSample>();
			
			//iterate by sample
			for(regressionSample sample: set) {
				temp.add(new regressionSample(sample.value,oneHotEncode(sample.features,extraSpots,categoricalFeatures)));
			}
			hotEncoded.add(temp);
		}
		return hotEncoded;
	}
	
	/*
	 * Function to do one hot encoding
	 */
	public static double[] oneHotEncode(double[] sample, int extraSpots, int[][] categoricalFeatures) {
		double[] inputs = new double[sample.length+extraSpots];
		int nextIndex=0; //counter variable for where the next value is to be stored in the array
		
		//iterate over original features
		for(int feature=0;feature<sample.length;feature++) {
			int indexOf = arrayContains(categoricalFeatures[0],feature);
			//if current feature is not categorical, do nothing
			if(indexOf==-1) {
				inputs[nextIndex]=sample[feature];
				nextIndex++;
			}
			//if current feature is categorical, hot encode it
			else {
				//this assumes that the categories range from 0 to X, and that X+1 is the number of categories specified in categoricalFeatures[1]
				for(int value=0;value<categoricalFeatures[1][indexOf];value++) {
					//add a 1 where the value/category number matches the category of the sample, and 0's elsewhere
					if(value==(int)sample[feature]) {
						inputs[nextIndex]=1.0;
						nextIndex++;
					}
					else {
						inputs[nextIndex]=0.0;
						nextIndex++;
					}
				}
			}
		}
		return inputs;
	}
	
	/*
	 * Method to copy a 2D array
	 */
	public static double[][] clone(double[][] input){
		double[][] clone = new double[input.length][];
		for(int index1=0;index1<input.length;index1++) {
			clone[index1]=new double[input[index1].length];
			for(int index2=0;index2<input[index1].length;index2++) {
				clone[index1][index2] = input[index1][index2];
			}
		}
		return clone;
	}
	
	/*
	 * Method to calculate 0-1 loss between predictions and real data
	 */
	public static double calc01Loss(String[] predictions, ArrayList<classificationSample> actual) {
		int errors = 0;
		for(int index=0;index<predictions.length;index++) {
			if(!predictions[index].equals(actual.get(index).classification)) {
				errors++;
			}
		}
		return (double) errors/predictions.length;
	}
	
	/*
	 * Method to calculate mean absolute error between predictions and real data
	 */
	public static double calcMAE(double[] predictions, ArrayList<regressionSample> actual) {
		double absError = 0;
		for(int index=0;index<predictions.length;index++) {
			absError += Math.abs(predictions[index]-actual.get(index).value);
		}
		return (double) absError/predictions.length;
	}
}
