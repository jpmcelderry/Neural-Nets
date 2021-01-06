
import java.util.*;
import java.io.*;

public class ML5 {

	public static void main(String[] args) throws Exception {
		//create arraylists
		ArrayList<ArrayList<classificationSample>> brca = new ArrayList<ArrayList<classificationSample>>();
		ArrayList<ArrayList<classificationSample>> glass = new ArrayList<ArrayList<classificationSample>>();		
		ArrayList<ArrayList<classificationSample>> soybean = new ArrayList<ArrayList<classificationSample>>();
		ArrayList<ArrayList<regressionSample>> abalone = new ArrayList<ArrayList<regressionSample>>();		
		ArrayList<ArrayList<regressionSample>> machines = new ArrayList<ArrayList<regressionSample>>();
		ArrayList<ArrayList<regressionSample>> fires = new ArrayList<ArrayList<regressionSample>>();
		for(String file:args) {
			switch(file) {
			case "breast-cancer-wisconsin.data":
				brca = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(brca,new String[] {"NORMAL","MALIGNANT"},new int[][]{{0,1,2,3,4,5,6,7,8},{10,10,10,10,10,10,10,10,10}},new int[][]{{8},{8,5}},0.02,true,file);
				break;
			case "glass.data":
				glass = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(glass,new String[] {"1","2","3","4","5","6","7"},new int[][]{{}},new int[][]{{100},{4,4}},0.02,true,file);
				break;
			case "soybean-small.data":
				soybean = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(soybean,new String[] {"D1","D2","D3","D4"},new int[][]{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34},
					{7,2,3,3,2,4,4,3,3,3,2,2,3,3,3,2,2,3,2,2,4,4,2,3,2,3,2,4,5,2,2,2,2,2,3}}, new int[][]{{10},{4,4}},0.5,true,file);
				break;
			case "abalone.data":
				abalone = partitionData.partitionRegressionData(readData.readRegressionData(file));
				fiveFoldRegress(abalone,new int[][]{{0},{3}},new int[][]{{20},{20,10}},.01,true,file);
				break;
			case "machine.data":
				machines = partitionData.partitionRegressionData(readData.readRegressionData(file));
				fiveFoldRegress(machines,new int[][]{{0},{30}},new int[][]{{200},{200,100}},0.001,true,file);
				break;
			case "forestfires.csv":
				fires = partitionData.partitionRegressionData(readData.readRegressionData(file));
				fiveFoldRegress(fires,new int[][]{{2,3},{12,7}},new int[][]{{200},{200,100}},0.001,true,file);
				break;
			}
		}
	}
	
	/*
	 * Driver method to classify an input dataset with five-fold classification
	 * Most input variables are self-explanatory, the categorical features 2d array must include two arrays: the first is the index of any categorical variables and the second is the
	 * number of categories possible (this can't be simply inferred in the case that a categorical feature is used late in a tree where few samples remain)
	 */
	public static void fiveFoldClassify(ArrayList<ArrayList<classificationSample>> samples, String[] classes, int[][] categoricalFeatures, int[][] hiddenNodes, double n, boolean printOutput, String fileName) throws IOException {
		BufferedWriter buffWriter = null;
		try{
			buffWriter = new BufferedWriter(new FileWriter(fileName + ".out.txt"));
			double[] zeroLayerError = new double[5];
			double[] oneLayerError = new double[5];
			double[] twoLayerError = new double[5];

			//z-score transform the data
			samples = helpers.zScoreTransformC(samples,categoricalFeatures);
			if(categoricalFeatures.length>1) {
				samples = helpers.oneHotEncodeC(samples,categoricalFeatures);
			}
			
			//iterate for all 5 folds
			for (int holdOut=0;holdOut<5;holdOut++) {
				ArrayList<classificationSample> trainingData = new ArrayList<classificationSample>();
				for(int trainingFold=0;trainingFold<5;trainingFold++) {
					if(trainingFold != holdOut) {
						for(classificationSample sample: samples.get(trainingFold)) {
							trainingData.add(sample);
						}
					}
				}
				//boolean to determine whether to print output
				boolean print=false;
				if(holdOut==0&&printOutput) {
					print=true;
				}
				
				//Zero Layer NN
				ffNeuralNet zeroLayers = new ffNeuralNet();
				zeroLayers.trainClassification(trainingData,new int[]{},classes,n,print);
				String[] results = zeroLayers.classify(samples.get(holdOut));
				if(print) {
					buffWriter.write("------ZERO LAYER RESULTS-------\n");
					for(int index=0;index<results.length;index++) {
						buffWriter.write("PREDICTED: " + results[index] + ", ACTUAL: " + samples.get(holdOut).get(index).classification + "\n");
					}
				}
				zeroLayerError[holdOut] = helpers.calc01Loss(results,samples.get(holdOut));
				
				//One Layer NN
				ffNeuralNet oneLayer = new ffNeuralNet();
				oneLayer.trainClassification(trainingData,hiddenNodes[0],classes,n,print);
				results = oneLayer.classify(samples.get(holdOut));
				if(print) {
					buffWriter.write("------ONE LAYER RESULTS-------\n");
					for(int index=0;index<results.length;index++) {
						buffWriter.write("PREDICTED: " + results[index] + ", ACTUAL: " + samples.get(holdOut).get(index).classification + "\n");
					}
				}
				oneLayerError[holdOut] = helpers.calc01Loss(results,samples.get(holdOut));
				
				//Two Layer NN
				ffNeuralNet twoLayers = new ffNeuralNet();
				twoLayers.trainClassification(trainingData,hiddenNodes[1],classes,n,print);
				results = twoLayers.classify(samples.get(holdOut));
				if(print) {
					buffWriter.write("------TWO LAYER RESULTS-------\n");
					for(int index=0;index<results.length;index++) {
						buffWriter.write("PREDICTED: " + results[index] + ", ACTUAL: " + samples.get(holdOut).get(index).classification + "\n");
					}
				}
				twoLayerError[holdOut] = helpers.calc01Loss(results,samples.get(holdOut));
				
				buffWriter.write("--------------HOLD OUT SET: " + (holdOut+1) + "---------------\n");
				buffWriter.write("Zero Hidden Layers Error %: " + (zeroLayerError[holdOut]*100) + "\n");
				buffWriter.write("One Hidden Layer Error %: " + (oneLayerError[holdOut]*100) + "\n");
				buffWriter.write("Two Hidden Layers Error %: " + (twoLayerError[holdOut]*100) + "\n\n");
			}
			double zeroErrorAvg = 0;
			double oneErrorAvg = 0;
			double twoErrorAvg = 0;
			for(int index=0;index<5;index++) {
				zeroErrorAvg += zeroLayerError[index]/5.0;
				oneErrorAvg += oneLayerError[index]/5.0;
				twoErrorAvg += twoLayerError[index]/5.0;
			}
			buffWriter.write("--------------AVERAGE PERFORMANCE---------------\n");
			buffWriter.write("Zero Hidden Layer Error %: " + (zeroErrorAvg*100) + "\nOne Hidden Layer Error %: " + (oneErrorAvg*100) + "\nTwo Hidden Layer Error %: " + (twoErrorAvg*100) + "\n\n");
		}
		finally {
			if(buffWriter!=null) {buffWriter.close();}
		}
	}
	
	/*
	 * Driver method to classify an input dataset with five-fold classification
	 * Most input variables are self-explanatory, the categorical features 2d array must include two arrays: the first is the index of any categorical variables and the second is the
	 * number of categories possible (this can't be simply inferred in the case that a categorical feature is used late in a tree where few samples remain)
	 */
	public static void fiveFoldRegress(ArrayList<ArrayList<regressionSample>> samples, int[][] categoricalFeatures, int[][] hiddenNodes, double n, boolean printOutput, String fileName) throws IOException {
		BufferedWriter buffWriter = null;
		try{
			buffWriter = new BufferedWriter(new FileWriter(fileName + ".out.txt"));
			double[] zeroLayerError = new double[5];
			double[] oneLayerError = new double[5];
			double[] twoLayerError = new double[5];
			
			//z-score transform the data
			samples = helpers.zScoreTransformR(samples,categoricalFeatures);
			if(categoricalFeatures.length>1) {
				samples = helpers.oneHotEncodeR(samples,categoricalFeatures);
			}
			
			//iterate for all 5 folds
			for (int holdOut=0;holdOut<5;holdOut++) {
				ArrayList<regressionSample> trainingData = new ArrayList<regressionSample>();
				for(int trainingFold=0;trainingFold<5;trainingFold++) {
					if(trainingFold != holdOut) {
						for(regressionSample sample: samples.get(trainingFold)) {
							trainingData.add(sample);
						}
					}
				}
				boolean print=false;
				if(holdOut==0&&printOutput) {
					print=true;
				}
				
				//Zero Layer NN
				ffNeuralNet zeroLayers = new ffNeuralNet();
				zeroLayers.trainRegression(trainingData,new int[]{},n,print);
				double[] results = zeroLayers.regress(samples.get(holdOut));
				if(print) {
					buffWriter.write("------ZERO LAYER RESULTS-------\n");
					for(int index=0;index<results.length;index++) {
						buffWriter.write("PREDICTED: " + results[index] + ", ACTUAL: " + samples.get(holdOut).get(index).value + "\n");
					}
				}
				zeroLayerError[holdOut]=helpers.calcMAE(results,samples.get(holdOut));
				
				//One Layer NN
				ffNeuralNet oneLayer = new ffNeuralNet();
				oneLayer.trainRegression(trainingData,hiddenNodes[0],n,print);
				results = oneLayer.regress(samples.get(holdOut));
				if(print) {
					buffWriter.write("------ONE LAYER RESULTS-------\n");
					for(int index=0;index<results.length;index++) {
						buffWriter.write("PREDICTED: " + results[index] + ", ACTUAL: " + samples.get(holdOut).get(index).value + "\n");
					}
				}
				oneLayerError[holdOut]=helpers.calcMAE(results,samples.get(holdOut));
				
				//Two Layer NN
				ffNeuralNet twoLayers = new ffNeuralNet();
				twoLayers.trainRegression(trainingData,hiddenNodes[1],n,print);
				results = twoLayers.regress(samples.get(holdOut));
				if(print) {
					buffWriter.write("------TWO LAYER RESULTS-------\n");
					for(int index=0;index<results.length;index++) {
						buffWriter.write("PREDICTED: " + results[index] + ", ACTUAL: " + samples.get(holdOut).get(index).value + "\n");
					}
				}
				twoLayerError[holdOut]=helpers.calcMAE(results,samples.get(holdOut));
				
				buffWriter.write("--------------HOLD OUT SET: " + (holdOut+1) + "---------------\n");
				buffWriter.write("Zero Hidden Layers Mean Aboslute Error: " + zeroLayerError[holdOut] + "\n");
				buffWriter.write("One Hidden Layer Mean Aboslute Error: " + oneLayerError[holdOut] + "\n");
				buffWriter.write("Two Hidden Layers Mean Aboslute Error: " + twoLayerError[holdOut] + "\n\n");	
			}
			double zeroErrorAvg = 0;
			double oneErrorAvg = 0;
			double twoErrorAvg = 0;
			for(int index=0;index<5;index++) {
				zeroErrorAvg += zeroLayerError[index]/5.0;
				oneErrorAvg += oneLayerError[index]/5.0;
				twoErrorAvg += twoLayerError[index]/5.0;
			}
			buffWriter.write("--------------AVERAGE PERFORMANCE---------------\n");
			buffWriter.write("Zero Hidden Layers Mean Aboslute Error: " + zeroErrorAvg + "\nOne Hidden Layer Mean Aboslute Error: " + oneErrorAvg + "\nTwo Hidden Layers Mean Aboslute Error: " + twoErrorAvg + "\n\n");
		}
		finally {
			if(buffWriter!=null) {buffWriter.close();}
		}
	}
}
