import java.util.*;

public class ffNeuralNet {
	
	double[][][] weights;
	double[][] bias;
	String[] classIndices = null;

	public void setup(int numFeatures, int[] hiddenNodes, String[] classes) {
		this.weights = new double[hiddenNodes.length+1][][];
		int numClasses;
		//Track how many classes there are, which is useful for creating the last layer of the network
		if(classes.length>0) {
			numClasses = classes.length;
			classIndices=classes;
		}
		//create 1 node if this is a regression problem
		else {
			numClasses = 1;
		}
		//code for creating weights matrix with hidden layers
		if(weights.length>1) {
			//code for creating weight matrix entering first hidden layer and leaving last hidden layer
			weights[0]=new double[hiddenNodes[0]][numFeatures];
			weights[weights.length-1] = new double[numClasses][hiddenNodes[hiddenNodes.length-1]];
			//If there is just one hidden layer, this code won't execute
			for(int hiddenLayer=1;hiddenLayer<hiddenNodes.length;hiddenLayer++) {
				weights[hiddenLayer] = new double[hiddenNodes[hiddenLayer]][hiddenNodes[hiddenLayer-1]];
			}
		}
		//code for creating weights matrix with 0 hidden layers
		else {
			weights[0] = new double[numClasses][numFeatures];
		}
		//create a bias value for each node beyond the input layer
		bias = new double[weights.length+1][];
		for(int layer=1;layer<bias.length;layer++) {
			bias[layer] = new double[weights[layer-1].length];
			for(int biasIndex=0;biasIndex<bias[layer].length;biasIndex++) {
				bias[layer][biasIndex]= (Math.random() * .02)-.01;
			}
		}
		//instantiate each weight to a small,random value
		for(int layer=0;layer<weights.length;layer++) {
			for(int node=0;node<weights[layer].length;node++) {
				for(int weight=0;weight<weights[layer][node].length;weight++) {
					weights[layer][node][weight] = (Math.random() * .02)-.01;
				}
			}
		}
	}
	
	/*
	 * Code to set up training for a classification problem
	 */
	public void trainClassification(ArrayList<classificationSample> samples, int[] hiddenNodes, String[] classes, double n, boolean print) {
		setup(samples.get(0).features.length,hiddenNodes,classes);
		int iterations = 0;
		while(iterations<500*(hiddenNodes.length+1)) {
			Collections.shuffle(samples);
			for(int index=0;index<samples.size();index++) {
				classificationSample sample=samples.get(index);
				//feed forward
				double[][] networkState = feedForward(sample.features,true);
				//print network state on first sample of first epoch
				if(print && iterations==0 && index==0) {
					printNetwork(networkState);
				}
				double[][] deltas = new double[networkState.length][];
				for(int layer=0;layer<networkState.length;layer++) {
					deltas[layer] = new double[networkState[layer].length];
				}
				//Determine what the correct class index is
				int correctClass=-1;
				for(int classIndex=0;classIndex<classIndices.length;classIndex++) {
					if(sample.classification.equals(classIndices[classIndex])) {
						correctClass=classIndex;
					}
				}
				int lastLayer = deltas.length-1;
				//Determine delta for the last layer
				for(int outputNode=0;outputNode<deltas[lastLayer].length;outputNode++) {
					double output = networkState[lastLayer][outputNode];
					if(correctClass==outputNode) {
						deltas[lastLayer][outputNode]= (1.0-output)*output*(1.0-output);
					}
					else {
						deltas[lastLayer][outputNode]= (0.0-output)*output*(1.0-output);
					}
				}
				//Print gradient
				if(print && iterations==0 && index==0) {
					System.out.println("GRADIENTS AT OUTPUT LAYER:");
					for(double error:deltas[lastLayer]) {
						System.out.print(error + "\t");
					}
					System.out.println("\n");
				}
				//Backpropagate through the network
				if(print && iterations==0 && index==0) {
					backprop(networkState,deltas,n,true);
				}
				else {
					backprop(networkState,deltas,n,false);
				}
			}
			iterations++;
		}
	}
	
	/*
	 * Code to set up training for a regression problem
	 */
	public void trainRegression(ArrayList<regressionSample> samples, int[] hiddenNodes, double n, boolean print) {
		setup(samples.get(0).features.length,hiddenNodes,new String[]{});
		
		int iterations = 0;
		while(iterations<500*(hiddenNodes.length+1)) {
			Collections.shuffle(samples);
			for(int index=0;index<samples.size();index++) {
				regressionSample sample = samples.get(index);
				//feed forward
				double[][] networkState = feedForward(sample.features,false);
				//print network state on first sample of first epoch
				if(print && iterations==0 && index==0) {
					printNetwork(networkState);
				}
				double[][] deltas = new double[networkState.length][];
				for(int layer=0;layer<networkState.length;layer++) {
					deltas[layer] = new double[networkState[layer].length];
				}
				int lastLayer = deltas.length-1;
				//Determine delta for the last layer
				double output = networkState[lastLayer][0];
				deltas[lastLayer][0]= (sample.value-output);
				//Print the gradient
				if(print && iterations==0 && index==0) {
					System.out.println("GRADIENTS AT OUTPUT LAYER:");
					for(double error:deltas[lastLayer]) {
						System.out.print(error + "\t");
					}
					System.out.println("\n");
				}
				//Backpropagate through the network
				if(print && iterations==0 && index==0) {
					backprop(networkState,deltas,n,true);
				}
				else {
					backprop(networkState,deltas,n,false);
				}
			}
			iterations++;
		}
	}
	
	/*
	 * Code to feed forward, returns the state of the network
	 */
	public double[][] feedForward(double[] inputs, boolean classify) {
		double[][] nodeVals = new double[weights.length+1][];
		nodeVals[0] = inputs;
		for(int layer=1;layer<nodeVals.length;layer++) {
			//instantiate array to store the node values
			nodeVals[layer] = new double[weights[layer-1].length];
			for(int node=0;node<nodeVals[layer].length;node++){
				double sum = 0;
				//calculate linear sum
				for(int input=0;input<weights[layer-1][node].length;input++) {
					sum += weights[layer-1][node][input]*nodeVals[layer-1][input];
				}
				//add bias
				sum+=bias[layer][node];
				//use logistic function (or don't on the output layer for a regression problem)
				if(layer<nodeVals.length-1 || classify) {
					nodeVals[layer][node]=1.0/(1.0+Math.exp(-1.0*sum));
				}
				else {
					nodeVals[layer][node]=sum;
				}
			}
		}
		//return the values of each node 
		return nodeVals;
	}
	
	/*
	 * Code to perform backpropagation by calculating delta for each node and then applying updates to each weight by n*delta*input
	 */
	public void backprop(double[][] networkState,double[][] deltas, double n, boolean print) {
		//Calculate dErr/dnetj for each node in the network starting at second last layer
		for(int layer=deltas.length-2;layer>0;layer--) {
			for(int i=0;i<deltas[layer].length;i++) {
				for(int j=0;j<deltas[layer+1].length;j++) {
					deltas[layer][i] += deltas[layer+1][j]*weights[layer][j][i]*networkState[layer][i]*(1.0-networkState[layer][i]);
				}
			}
		}
		//calculate and apply the weight updates
		for(int layer=0;layer<weights.length;layer++) {
			if(print) {
				System.out.print("WEIGHT UPDATES, LAYER " + layer + " TO " + (layer+1) + ":\n");
			}
			for(int node=0;node<weights[layer].length;node++) {
				if(print) {
					System.out.print("[");
				}
				for(int weight=0;weight<weights[layer][node].length;weight++) {
					double weightUpdate = n*deltas[layer+1][node]*networkState[layer][weight];
					if(print) {
						System.out.print(weightUpdate + "\t");
					}
					weights[layer][node][weight] += weightUpdate;
				}
				if(print) {
					System.out.print("]\n");
				}
			}
		}
		for(int layer=1;layer<bias.length;layer++) {
			for(int node=0;node<bias[layer].length;node++) {
				bias[layer][node]+=n*deltas[layer][node];
			}
		}
	}
	
	/*
	 * Code to detect the class prediction from the final state of the network
	 */
	public String detectClass(double[][] networkState){
		int outputLayer = networkState.length-1;
		int predictedClass = 0;
		for(int outputNode=1;outputNode<networkState[outputLayer].length;outputNode++) {
			if (networkState[outputLayer][outputNode]>networkState[outputLayer][predictedClass]) {
				predictedClass = outputNode;
			}
		}
		return classIndices[predictedClass];
	}
	
	/*
	 * Code to classify a sample set, returns an array of the predictions
	 */
	public String[] classify(ArrayList<classificationSample> samples) {
		String[] results = new String[samples.size()];
		for(int sampleNumber=0;sampleNumber<samples.size();sampleNumber++) {
			double[][] networkState = feedForward(samples.get(sampleNumber).features,true);
			results[sampleNumber]=detectClass(networkState);
		}
		return results;
	}
	
	/*
	 * Code to regress a sample set, returns an array of the predictions
	 */
	public double[] regress(ArrayList<regressionSample> samples) {
		double[] results = new double[samples.size()];
		for(int sampleNumber=0;sampleNumber<samples.size();sampleNumber++) {
			double[][] networkState = feedForward(samples.get(sampleNumber).features,false);
			int outputLayer = networkState.length-1;
			results[sampleNumber]=networkState[outputLayer][0];
		}
		return results;
	}
	
	/*
	 * Code to print the network state and weights matrix
	 */
	public void printNetwork(double[][] networkState) {
		for(int layer=0;layer<networkState.length;layer++) {
			if(layer==0) {
				System.out.print("INPUT LAYER:\n[ ");
			}
			else if(layer==networkState.length-1) {
				System.out.print("OUTPUT LAYER:\n[ ");
			}
			else {
				System.out.print("HIDDEN LAYER " + layer + ":\n[ ");
			}
			for(int node=0;node<networkState[layer].length;node++) {
				System.out.print((Math.round(networkState[layer][node]*1000.0)/1000.0) + "\t");
			}
			System.out.print("]\n");
			if(layer<weights.length) {	
				System.out.print("WEIGHTS, LAYER " + layer + " TO " + (layer+1) + ":\n");
				for(int node=0;node<weights[layer].length;node++) {
					System.out.print("[\t");
					for(int weight=0;weight<weights[layer][node].length;weight++) {
						System.out.print((Math.round(weights[layer][node][weight]*1000.0)/1000.0) + "\t");
					}
					System.out.print("]\n");
				}
			}
		}
	}
}
