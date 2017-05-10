package interface2;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.SMOreg;



/**
 * classifier ��� ���� <br/>
 * ��ī ��Ű�� �� GP, RBFR, SVM, DNN�� ��θ� ����
 *
 */
public class Keyword {
	public static String[] FUNCTION_DICTIONARY = { "GP", "RBFR", "SVM", "MLP","DL4J" };

	public static String PreClassifier = "weka.classifiers.";
	public static String PreFilter = "weka.filters.";

	public static String GP = "functions.GaussianProcesses";
	public static String RBFR = "functions.RBFRegressor";
	public static String SVM = "functions.SMOreg";
	public static String MLP = "functions.MLPRegressor";
	public static String DL4J = "functions.Dl4jMlpClassifier";

	private static Object Object;

	/**
	 * ����� classifier ���� �� �� classifier�� �⺻ �ɼ� ����
	 * @param function		classifier �̸�
	 * @param option		����� classifier ����
	 * @return
	 * @throws Exception
	 */
	public static Classifier getModel(String function,String option) throws Exception {
		String[] options = weka.core.Utils.splitOptions(option);
		if (function.equals("GP"))
			return AbstractClassifier.forName(PreClassifier + GP, options);
		else if (function.equals("SVM"))
			return AbstractClassifier.forName(PreClassifier + SVM,options);
		else if (function.equals("MLP"))
			return AbstractClassifier.forName(PreClassifier + MLP,options);
		else if (function.equals("RBFR"))
			return AbstractClassifier.forName(PreClassifier + RBFR,options);
		else if (function.equals("DL4J"))
			return AbstractClassifier.forName(PreClassifier + DL4J,options);
		else
			return null;
	}
}
