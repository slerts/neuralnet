package nn;

import java.util.Arrays;

public class TestHarness {

	// iris.csv
	//	1,0,0	Iris-setosa
	//	0,1,0	Iris-versicolor
	//	0,0,1	Iris-virginica
	
	// breast-cancer-data
	// 1,0	benign (formerly 2)
	// 0,1	malignant (formerly 4)
	
	// classes
	
	// constants
	
	static String DATAFILE = "bcw.csv";
	static double LEARNRATE = 0.5; // only for backprop
	static double MOMENTUM = 0.05; // only for backprop
	static int EPOCHS = 1000;
	static String ACTIVATION = "sig";
	static int HIDDENSIZE = 8;
	static int BIAS = 1;
	static double HOLDOUT = 0.8; // total amount of data to use for training, remainder is held for testing
	
	// variables
	static int allClassed = 0; // tracks number of classed input vectors during training
	static int correctTest = 0; // tracks number of correct classifications during testing
	
	// print array of ints
	public static void printMatrix(double[] matrix, int cols) {
		for (int col = 0; col < cols - BIAS; col++) {
			System.out.print(matrix[col] + " ");
		}
	}
	
	public static void backProp(NeuralNet nn) {
		// loops data set and feeds input vectors into network
				for (int row = 0; row < nn.trainingSize; row++) {

					String rightAns = "No";

					nn.loadInput(true, row); // load next vector from data set
					nn.fwdPass(); // pass vector through network
					nn.calcOutputGrad(); // calculate output error
					nn.bpWeightUpdate(); // backpropagate the error and update weights

					if (nn.isCorrect()) {
						allClassed++; // count vector as classified correctly
						rightAns = "Yes";
					}

					// write data to training log
//					nn.trainingLog.format("%-50s%-20s%-20s%-60s%n", Arrays.toString(nn.input), Arrays.toString(nn.output), rightAns, Arrays.toString(nn.outputActivated));

				}
	}
	
	public static void rProp(NeuralNet nn) {
		// loops data set and feeds input vectors into network
		for (int row = 0; row < nn.trainingSize; row++) {

			String rightAns = "No";

			nn.loadInput(true, row); // load next vector from data set
			nn.fwdPass(); // pass vector through network
			nn.rpOutputError();
			nn.calcOutputGrad(); // calculate output error
			nn.calcHiddenDeltas();
			nn.rpAccumGrad();
			nn.rpUpdateWeight ();


			if (nn.isCorrect()) {
				allClassed++; // count vector as classified correctly
				rightAns = "Yes";
			}

			// write data to training log
//			nn.trainingLog.format("%-50s%-20s%-20s%-60s%n", Arrays.toString(nn.input), Arrays.toString(nn.output), rightAns, Arrays.toString(nn.outputActivated));

		}
	}
	
	// main program fxn
	public static void main(String[] args) {
		
		// run for 30 iterations to for statistical purposes
//		for (int i = 0; i < 30; i++) {
			String expID = "test";
			NeuralNet nn = new NeuralNet(DATAFILE, LEARNRATE, MOMENTUM, ACTIVATION, HIDDENSIZE, HOLDOUT, BIAS, expID);
			allClassed = 0;
			correctTest = 0;
			/*
			 * Training
			 * 
			 */
			System.out.println("Training Network - Run#" );
			int ecnt = 0; // track epoch interval
			while (ecnt < EPOCHS && allClassed != nn.trainingSize) {
				allClassed = 0; // reset variable that tracks classified inputs

				// write training epoch count and table headers to log file
				//			nn.trainingLog.format("EPOCH: %s of %s%n", ecnt, EPOCHS);
				//			nn.trainingLog.format("%-50s%-20s%-20s%-60s%n","Input Vector", "Expected", "Correct", "Actual Output");

				// write initial weights to log file
				//			nn.writeWeightLogs();

				backProp(nn);

				double correctClass = (double)allClassed / nn.trainingSize * 100.0;
				nn.performanceLog.format("%f,", correctClass);
				//			nn.trainingLog.format("%n");
				nn.shuffle(nn.trainingSet, nn.trainExpOutput); // shuffle data set
				ecnt++;
			}
			System.out.println("Training Completed after " + ecnt + " epochs.");
			System.out.println("Training Parameters:\n\tLearning Rate: " + LEARNRATE + "\n\tMomentum: " + MOMENTUM + "\n\tHidden Nodes: " + HIDDENSIZE + "\n\tRandom Seed: " + nn.randomSeed);
			System.out.print("% Classified correctly: ");
			System.out.println((double)allClassed / nn.trainingSize * 100.0 + "%");

			/*
			 * Testing
			 * 
			 */
			System.out.println("\nRunning Test...");
			System.out.print("Input Vector\t\tExpected\t\tCorrect\t\t\tClass\n");
			nn.shuffle(nn.testingSet, nn.testExpOutput);// works
			for (int row = 0; row < nn.testingSize; row++) {
				nn.loadInput(false, row); // works
				nn.fwdPass(); // works
				printMatrix(nn.input, nn.inputSize);
				System.out.print("\t" + Arrays.toString(nn.output)); //expected output
				if (nn.isCorrect()) {
					System.out.print("\t\tYes\t");
					correctTest++;
				} else {
					System.out.print("\t\tNo\t");
				}
				System.out.print("\t\t" + Arrays.toString(nn.outputActivated) + "\n"); // classificiation
			}
			System.out.print("Percent Correct: ");
			System.out.println((float)correctTest / nn.testingSize * 100 + "%");

			nn.closeLogs();
		
	 }
}
