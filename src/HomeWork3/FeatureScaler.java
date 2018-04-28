package HomeWork3;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public Instances scaleData (Instances instances) {
		Instances scaled = new Instances( instances, 0);
		Instance newInst = null;


		double [] stdDevArr = new double [instances.numAttributes()];
		double [] meanArr  = new double [instances.numAttributes()];

		for (int i = 0; i < instances.numAttributes(); i++) {

			stdDevArr[i] = StdDev(instances, i);
			meanArr[i] = mean(instances, i);

		}


		for (int i = 0; i < instances.numInstances(); i++) {
			newInst = filter(instances.get(i), stdDevArr, meanArr);
			scaled.add(newInst);
		}


		return scaled;
	}

	private Instance filter (Instance currIns, double[] std, double[] mean) {
		double [] currInsVal = new double [currIns.numAttributes()];
		for (int i = 0; i < currIns.numAttributes(); i++) {
			if (std[i] != -1) {
				currInsVal[i] = (currIns.value(i) - mean[i] ) / std[i];
			} else {
				currInsVal[i] = currIns.value(i);
			}
		}
		return currIns.copy(currInsVal);
	}

	private double StdDev (Instances instances, int numAttr) {
		return instances.attribute(numAttr).isNumeric() ? Math.sqrt(instances.variance(numAttr)) : -1;
	}

	private double mean (Instances instances, int numAttr) {
		double sum = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			sum += instances.get(i).value(numAttr);
		}

		return sum / instances.numInstances();
	}
}