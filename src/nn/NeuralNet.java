package nn;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Formatter;
import java.util.Iterator;
import java.util.Random;

import com.opencsv.CSVReader;

/*
 * Assignment 1 - Neural Net
 * 
 * Neural net implemented with a series of matrices. Activation functions include sigmoid and tanh.
 * However, issues appear with tanh as training never reaches about 60% and often stays below 50%.
 * Rprop and Normal backprop are implemented. However, issues with Rprop implementation also
 * prevent proper learning from occuring. No report was produced for this assignment as proper implementation
 * of all aspect was no achieved and therefore statistical analysis seems moot. Analysis could have been done soley
 * on backpropagation with groups split based on parameter types. However, most time was devoted to working out
 * bugs and didn't permit for proper report to be produced.
 * 
 * regular backprop was tested on both iris and bcw data sets and is capable of calssifying with great accuracy.
 * parameters are controled through test harness and results printed to the screen.
 */

public class NeuralNet {
	// network properties
	int bias;
	int hiddenSize;
	double learnRate;
	double momentum;
	String activation;

	// data input
	ArrayList<String[]> inputData = new ArrayList<>();
	int inputSize;
	int outputSize;
	int trainingSize;
	int testingSize;
	double[][] trainingSet;
	double[][] testingSet;
	double[][] trainExpOutput;
	double[][] testExpOutput;

	// matrices
	double[] input; // input loaded from file
	double[] output; // expected output loaded from file
	double[][] inputWeights; // synaptic weights for input to hidden layer
	double[][] hiddenWeights; // synaptic weights for hidden layer to output
	double[] hiddenActivated; // hidden layer outputs (weighted sums passed through sigmoid fxn)
	double[] hiddenErr; // error rates for synapses from hidden layer on output layer
	double[] outputActivated; // output layer outputs (weighted sums passed through sigmoid fxn)
	double[] outputErr; // output layer error

	// runtime variables
	long randomSeed = System.currentTimeMillis();
	Random rnd = new Random(randomSeed);
	boolean correct;

	// backprop specific variables
	double[][] inputMag; // magnitude change in weights for input to hidden layer
	double[][] hiddenMag; // magnitude change in weights for hidden layer to output

	// rProp specific variables
	double mserror = 0.0;
	double[][] inGrad; // magnitude change in weights for input to hidden layer
	double[][] hidGrad; // magnitude change in weights for hidden layer to output
	double[][] prevInGrad; // magnitude change in weights for input to hidden layer
	double[][] prevHidGrad; // magnitude change in weights for hidden layer to output
	double[][] prevInDelta; // magnitude change in weights for input to hidden layer
	double[][] prevHidDelta; // magnitude change in weights for hidden layer to output

	// I/O
	Formatter trainingLog;
	Formatter inputWeightLog;
	Formatter hiddenWeightLog;
	Formatter performanceLog;

	public NeuralNet(String dataFile, double learnRate, double momentum, String activation, int hiddenSize, double holdOut, int bias, String expID) {
		this.learnRate = learnRate;
		this.momentum = momentum;
		this.activation = activation;
		this.hiddenSize = hiddenSize;
		this.bias = bias;

		prepData(dataFile, holdOut); //initiate data set and input matrices
		configNet(); // create neural net structure
		openLogs(expID); // open the log files 
		shuffle(trainingSet, trainExpOutput); // shuffle the data set
		initWeights(); // set initial synaptic weights 
	}

	public void configNet() {
		// add bias to hidden layers
		inputSize += bias;
		hiddenSize += bias;

		// initialize matrices based on user input
		input = new double[inputSize];
		output = new double[outputSize];
		inputWeights = new double[inputSize][hiddenSize];
		hiddenWeights = new double[hiddenSize][outputSize];
		inputMag = new double[inputSize][hiddenSize];
		hiddenMag = new double[hiddenSize][outputSize];
		hiddenActivated = new double[hiddenSize];
		hiddenErr = new double[hiddenSize];
		outputActivated = new double[outputSize];
		outputErr = new double[outputSize];

		// rProp
		inGrad = new double[inputSize][hiddenSize];
		hidGrad = new double[hiddenSize][outputSize];
		prevInGrad = new double[inputSize][hiddenSize];
		prevHidGrad = new double[hiddenSize][outputSize];
		prevInDelta = new double[inputSize][hiddenSize];
		prevHidDelta = new double[hiddenSize][outputSize];

		//set bias values to 1
		input[inputSize - 1] = 1;
		hiddenActivated[hiddenSize - 1] = 1;
	}



	public void initWeights() {
		// initialize weights for synapses from input to hidden layer
		for (int row = 0; row < inputSize; row++) {
			for (int col = 0; col < hiddenSize; col++) {
				inputWeights[row][col] = rnd.nextDouble() - 0.5;
				inputMag[row][col] = 0;
			}
		}
		// initialize weights for synapses from hidden layer to output layer
		for (int row = 0; row < hiddenSize; row++) {
			for (int col = 0; col < outputSize; col++) {
				hiddenWeights[row][col] = rnd.nextDouble() - 0.5;
				hiddenMag[row][col] = 0;
			}
		}
	}

	public void loadInput(boolean train, int row) {
		// load next row from dataSet[][] and its expected output
		if (train) {
			for (int col = 0; col < (inputSize - bias); col++) {
				input[col] = trainingSet[row][col];
			}
			for (int col = 0; col < outputSize; col++) {
				output[col] = trainExpOutput[row][col];
			}
		} else {
			for (int col = 0; col < (inputSize - bias); col++) {
				input[col] = testingSet[row][col];
			}
			for (int col = 0; col < outputSize; col++) {
				output[col] = testExpOutput[row][col];
			}
		}
	}

	public double activationFxn(double sum, String fxnFlag) {
		// fxnFlag denotes type of activation fxn
		// sig = sigmoid
		// tan = tanh
		if (fxnFlag == "sig") {
			// sigmoid function for scaling neuron output between 0 and 1
			sum = -1.0 * sum;
			return (1 / (1 + Math.exp(sum)));
		} else if (fxnFlag == "tan") {
			// tanh function for scaling neuron output between -1 and 1
			// derivative is (1-tanh^2)
			double negSum = -1.0 * sum;
			return ((Math.exp(sum) - Math.exp(negSum)) / (Math.exp(sum) + Math.exp(negSum)));
		} else {
			System.err.println("Error: No Activation Function declared");
			return Double.MIN_VALUE;
		}
	}

	public double derivativeFxn(double value, String fxnFlag) {
		// derivative will vary depending on activation function
		// sig = sigmoid
		// tan = tanh
		if (fxnFlag == "sig") {
			// sigmoid function for scaling neuron output between 0 and 1
			return value * (1.0 - value);
		} else if (fxnFlag == "tan") {
			// tanh function for scaling neuron output between -1 and 1
			// derivative is (1-tanh^2)
			return 1 - Math.pow(activationFxn(value, activation), 2);
		} else {
			System.err.println("Error: No Activation Function declared");
			return Double.MIN_VALUE;
		}
	}

	public void fwdPass() {
		double[] hiddenSum = new double[hiddenSize]; // weighted sums of inputs to hidden layer
		double[] outputSum = new double[outputSize]; // weighted sums of hidden layer inputs to output nodes
		// pass input to hidden layer
		for (int col = 0; col < hiddenSize; col++) {
			for (int row = 0; row < inputSize; row++) {
				hiddenSum[col] += input[row] * inputWeights[row][col];
			}
			// pass weighted sum through sigmoid fxn
			hiddenActivated[col] = activationFxn(hiddenSum[col], activation);
		}
		// pass hidden layer to output
		for (int col = 0; col < outputSize; col++) {
			for (int row = 0; row < hiddenSize; row++) {
				outputSum[col] += hiddenActivated[row] * hiddenWeights[row][col];
			}
			// pass weighted sum through sigmoid fxn
			outputActivated[col] = activationFxn(outputSum[col], activation);
		}
	}

	public void calcOutputGrad() {
		// check if classified output is correct
//		isCorrect();
		// calc error for all
		for (int i = 0; i < outputSize; i++) {
			outputErr[i] = (output[i] - outputActivated[i]) * derivativeFxn((outputActivated[i]), activation);
		}
	}

	public void calcHiddenDeltas() {
		hiddenErr = new double[hiddenSize]; // reset hidden error to 0
		// backprop error to hidden layer nodes
		for (int row = 0; row < hiddenSize; row++) {
			double error = 0.0;
			for (int col = 0; col < outputSize; col++) {
				error += ((outputErr[col] * hiddenWeights[row][col]));
			}
			hiddenErr[row] = error * derivativeFxn(hiddenActivated[row], activation);
		}
	}

	public boolean isCorrect() {
		int expMax = 0;
		int actMax = 0;
		outputErr = new double[outputSize];
		for (int i = 1; i < output.length; i++) {
			if (output[i] > output[expMax])
				expMax = i;
			if (outputActivated[i] > outputActivated[actMax])
				actMax = i;
		}
		if (expMax == actMax) {
			return true;
		}
		return false;
	}



	/*
	 * Backprop specific methods
	 * 
	 */

	public void bpWeightUpdate() {
		// calculate back errors
		calcHiddenDeltas();

		// update weights for synapses from input layer onto hidden layer
		for (int col = 0; col < hiddenSize; col++) {
			for (int row = 0; row < inputSize; row++) {
				inputWeights[row][col] += (learnRate * hiddenErr[col] * input[row] + (inputMag[row][col] * momentum));
				inputMag[row][col] = hiddenErr[col] * input[row];
			}
		}
		// update weights for synapses from output layer onto hidden layer
		for (int col = 0; col < outputSize; col++) {
			for (int row = 0; row < hiddenSize; row++) {
				hiddenWeights[row][col] += (learnRate * outputErr[col] * hiddenActivated[row] + (hiddenMag[row][col] * momentum));
				hiddenMag[row][col] = outputErr[col] * hiddenActivated[row];
			}
		}
	}

	/*
	 * rProp specific methods
	 * 
	 */

	public void rpOutputError() {
		// calc error for all
		for (int i = 0; i < outputSize; i++) {
			mserror += 0.5 * Math.pow(output[i] - outputActivated[i], 2);
		}
	}

	public void rpAccumGrad() {
		// accumulate hidden gradients
		for (int i = 0; i < hiddenSize; i++) {
			for (int j = 0; j < outputSize; j++) {
				hidGrad[i][j] += outputErr[j] * hiddenActivated[i]; 
			}
		}
		// accumulate input gradients
		for (int i = 0; i < inputSize; i++) {
			for (int j = 0; j < hiddenSize; j++) {
				inGrad[i][j] += hiddenErr[j] * input[i]; 
			}
		}
	}

	public void rpUpdateWeight () {
		double npos = 1.2;
		double nneg = 0.5;
		double dmin = 0.000001;
		double dmax = 50.0;

		// input weights
		for (int i = 0; i < inputSize; i++) {
			for (int j = 0; j < hiddenSize; j++) {
				double delta;
				if (inGrad[i][j] * prevInGrad[i][j] > 0) {
					delta = Math.min(prevInGrad[i][j] * npos, dmax);
					inputWeights[i][j] += Math.abs(delta);
				} else if (inGrad[i][j] * prevInGrad[i][j] < 0) {
					delta = Math.max(prevInGrad[i][j] * nneg, dmin);
					inputWeights[i][j] -= prevInGrad[i][j];
					inGrad[i][j] = 0;
				} else {
					delta = prevInDelta[i][j];
					double absDelta = Math.abs(delta);
					inputWeights[i][j] += absDelta;
				}
				prevInDelta[i][j] = delta;
				prevInGrad[i][j] = inGrad[i][j];
			}
		}
		// hidden weights
		for (int i = 0; i < hiddenSize; i++) {
			for (int j = 0; j < outputSize; j++) {
				double delta;
				if (hidGrad[i][j] * prevHidGrad[i][j] > 0) {
					delta = Math.min(prevHidGrad[i][j] * npos, dmax);
					hiddenWeights[i][j] += Math.abs(delta);
				} else if (hidGrad[i][j] * prevHidGrad[i][j] < 0) {
					delta = Math.max(prevHidGrad[i][j] * nneg, dmin);
					hiddenWeights[i][j] -= prevHidGrad[i][j] ;
					hidGrad[i][j] = 0;
				} else {
					delta = prevHidDelta[i][j];
					double absDelta = Math.abs(delta);
					hiddenWeights[i][j] += absDelta;
				}
				prevHidDelta[i][j] = delta;
				prevHidGrad[i][j] = hidGrad[i][j];
			}
		}
	}

	/*
	 * Data processing
	 * 
	 */
	public void shuffle(double[][] in, double[][] out) {
		// shuffles the dataSet to ensure random order is fed into neural net
		int size = in.length * (inputSize - bias);
		for (int i = size; i > 1; i--)
			swap(in, out, i - 1, rnd.nextInt(i));
	}

	public void swap(double[][] in, double[][] out, int i, int j) {
		// works with shuffle, swaps 2 rows in data set and corresponding expected output
		double[] tmp = in[i / (inputSize - bias)];
		in[i / (inputSize - bias)] = in[j / (inputSize - bias)];
		in[j / (inputSize - bias)] = tmp;

		tmp = out[i / (inputSize - bias)];
		out[i / (inputSize - bias)] = out[j / (inputSize - bias)];
		out[j / (inputSize - bias)] = tmp;
	}

	// prep data
	public void prepData(String file, double holdOut) {
		// read data from file
		CSVReader reader;
		try {
			reader = new CSVReader(new FileReader(file));
			String [] nextLine;
			while ((nextLine = reader.readNext()) != null) {
				inputData.add(nextLine);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		String[] dataShape = inputData.remove(0);
		trainingSize = (int) (inputData.size() * holdOut);
		testingSize = inputData.size() - trainingSize;
		inputSize = Integer.parseInt(dataShape[0]);
		outputSize = Integer.parseInt(dataShape[1]);
		trainingSet = new double[trainingSize][inputSize];
		testingSet = new double[testingSize][inputSize];
		trainExpOutput = new double[trainingSize][outputSize];
		testExpOutput = new double[testingSize][outputSize];
		int i = 0;
		Collections.shuffle(inputData, rnd);
		for (Iterator<String[]> iter = inputData.iterator(); iter.hasNext();) {
			String[] strData = (String[]) iter.next();
			for (int j = 0; j < strData.length; j++) {
				if (j < inputSize) {
					if (i < trainingSize) {
						trainingSet[i][j] = Double.parseDouble(strData[j]);
					} else {
						testingSet[i-trainingSize][j] = Double.parseDouble(strData[j]);
					}
				} else {
					if (i < trainingSize) {
						trainExpOutput[i][j-inputSize] = Double.parseDouble(strData[j]);
					} else {
						testExpOutput[i-trainingSize][j-inputSize] = Double.parseDouble(strData[j]);
					}
				}
			}
			i++;
		}
	}

	/*
	 * File I/O
	 */
	public void openLogs(String expID) {
		// open log files, will overwrite file if already exists
//		try {
//			trainingLog = new Formatter("logs/trainingLog.txt");
//		} catch (Exception e) {
//			System.out.println("Can't open logs/trainingLog.txt");
//		}
//
//		try {
//			inputWeightLog = new Formatter("logs/inputWeightLog.csv");
//		} catch (Exception e) {
//			System.out.println("Can't open logs/inputWeightLog.csv");
//		}
//
//		try {
//			hiddenWeightLog = new Formatter("logs/hiddenWeightLog.csv");
//		} catch (Exception e) {
//			System.out.println("Can't open logs/hiddenWeightLog.csv");
//		}
		
		try {
			performanceLog = new Formatter("logs/performance-" + expID + ".csv");
		} catch (Exception e) {
			System.out.println("Can't open logs/performance-" + expID + ".csv");
		}
	}

	public void writeWeightLogs() {
		for (int i = 0; i < inputWeights.length; i++) {
			for (int j = 0; j < inputWeights[i].length; j++) {
				inputWeightLog.format("%f,", inputWeights[i][j]);
			}
			inputWeightLog.format("%n");
		}
		inputWeightLog.format("%n");

		for (int i = 0; i < hiddenWeights.length; i++) {
			for (int j = 0; j < hiddenWeights[i].length; j++) {
				hiddenWeightLog.format("%f,", hiddenWeights[i][j]);
			}
			hiddenWeightLog.format("%n");
		}
		hiddenWeightLog.format("%n");
	}

	public void closeLogs() {
		// close log files
//		trainingLog.close();
//		inputWeightLog.close();
//		hiddenWeightLog.close();
		performanceLog.close();
	}

}
