package interface2;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.evaluation.output.prediction.Null;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import jxl.Workbook;
import jxl.write.Label;
import jxl.write.NumberFormat;
import jxl.write.NumberFormats;
import jxl.write.WritableCellFormat;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;

/**
 * Weka API �������̽�
 */

public class WekaAPI {

static int sheetnum=0;
//엑셀파일 객체 생성
static WritableWorkbook workbook = null;

// 시트 객체 생성
static WritableSheet sheet = null;

// 셀 객체 생성
static Label label = null;

static jxl.write.Number number=null;

	/** the classifier used internally */
protected Classifier m_Classifier = null;

/** the filter to use */
protected Filter m_Filter = null;

/** the training file */
protected DataSource m_TrainingFile = null;

/** the training instances */
protected Instances m_Training = null;

/** for evaluating the classifier */
protected Evaluation m_Evaluation = null;

private String[] lastResult = null;

private boolean build = false;

/**
 * Weka API �Լ� �ʱ�ȭ
 * @param m_Classifier		����� classifier
 * @param m_Filter			����� ����
 * @param m_TrainingFile	�н��� ������ ����
 * @param m_Training		classifier ���� �ɼ�
 * @param m_Evaluation		classifier Evaluation
 * @param lastResult		���� �����
 * @param build  			���� ����
 */
public WekaAPI() {
	super();

	lastResult = new String[3];
}

/**
 * Weka API���� ����� �Լ� �ҷ�����
 */
public WekaAPI(WekaAPI src) {
	m_Classifier = src.m_Classifier;
	m_Filter = src.m_Filter;
	m_TrainingFile = src.m_TrainingFile;
	m_Training = src.m_Training;
	m_Evaluation = src.m_Evaluation;
	lastResult = src.lastResult;
	build = src.build;
}

/**
 * ����� classifier ����
 * @param modelName		��(classifier) �̸�
 * @param option		�� �ɼ�
 * @return				classifier�� ���������� ���õǸ� true ��ȯ
 * @throws Exception
 */
public boolean setModel(String modelName, String option) throws Exception {
	m_Classifier = null;
	for (String function : Keyword.FUNCTION_DICTIONARY) {
		if (modelName.equals(function)) {
			m_Classifier =Keyword.getModel(function, option);
			break;
		}
	}

	if (m_Classifier.equals(null))
		return false;

	return true;
}

/**
 * ����� ������ ���� ����
 * @param dataFile		�н� ������ ����
 * @return				�н� ������ ������ ���������� �ҷ����� true ��ȯ
 * @throws Exception
 */
public boolean setData(String dataFile) throws Exception {
	m_TrainingFile = new DataSource(dataFile);

	if (m_TrainingFile.equals(null))
		return false;

	return true;
}

/**
 * �н� �ɼ� ���� (�� �𵨺��� �ٸ�)
 * @return		�ɼ��� ���������� �ҷ����� true ��ȯ
 * @throws Exception
 */
public boolean setTraining(int indexNum) throws Exception {
	m_Training = m_TrainingFile.getDataSet();
	m_Training.setClassIndex(m_Training.numAttributes() - (indexNum+1));

	if(m_Training.equals(null))
		return false;

	return true;
}
/**
 * �� ����
 * @param modelName		classifier ����
 * @param option		classifier �ɼ�
 * @param dataFile		�н��� ������ ����
 * @return				���尡 ���������� ������ true ��ȯ
 * @throws Exception
 */
public boolean buildModel(String modelName, String option, String dataFile,int indexNum) throws Exception {
	boolean result = false;
	if(setModel(modelName, option) && setData(dataFile) && setTraining(indexNum))
		result = true;

	m_Classifier.buildClassifier(m_Training);
	build = true;

	return result;
}
/*public boolean buildModel(String modelName, String dataFile) throws Exception {
	      return buildModel(modelName, "", dataFile);
   }

public boolean buildModel(String modelName, String[] option, String dataFile) throws Exception {
      String oneLineOption = "";

  for(String s : option) {
     oneLineOption += s + " ";
      }

      return buildModel(modelName, oneLineOption, dataFile);
   }*/
/**
 * �����Ȳ üũ
 * @return
 */
public boolean isBuild() {
	return build;
}
/**
 * ����� ��������
 * @return
 */
public String[] getLastResult() {
	return lastResult;
}
/**
 * ������� �з����� ���� ����� �迭�� �Է�
 * @param modelString		��
 * @param detailString		���� ���
 * @param summaryString		���
 */
protected void setLastResult(String modelString, String detailString, String summaryString) {
	lastResult[0] = modelString;
	lastResult[1] = detailString;
	lastResult[2] = summaryString;
}

/**
 * CrossValidation ����
 * @param fold		���� ��
 * @param seed		�õ� ��
 * @return
 * @throws Exception
 */
public String[] runCrossValidation(int fold, int seed) throws Exception {
	if(!isBuild())
		return null;

	StringBuffer detail = new StringBuffer();
	/*CSV output = new CSV();
	output.setUseTab(true);*/
	PlainText output= new PlainText();
	output.setBuffer(detail);
	output.setAttributes("5");
	output.setHeader(new Instances(m_Training,0));
	output.setNumDecimals(5);
	Evaluation validation = new Evaluation(m_Training);
	validation.crossValidateModel(m_Classifier, m_Training, fold, m_Training.getRandomNumberGenerator(seed), output, null, true);

	setLastResult(m_Classifier.toString(), detail.toString(), validation.toSummaryString());

	return lastResult;
}

/**
 * TestSet ����
 * @param path		������ �ҽ� ��ġ
 * @return
 * @throws Exception
 */
public String[] runTestSet(String path,int indexNum) throws Exception {
	if(!isBuild())
		return null;

	DataSource testData = new DataSource(path);
	Instances testSet = testData.getDataSet();
	testSet.setClassIndex(testSet.numAttributes() - (indexNum+1));

	if(testSet.equals(null))
		return null;

	StringBuffer detail = new StringBuffer();
	CSV output = new CSV();
	output.setUseTab(true);
	/*PlainText output= new PlainText();*/
	output.setBuffer(detail);
	output.setAttributes("5");
	output.setNumDecimals(5);
	output.setHeader(new Instances(m_Training,0));
	Evaluation validation = new Evaluation(m_Training);


	validation.evaluateModel(m_Classifier, testSet, output);
	//validation.evaluateModel(m_Classifier,m_Training, output);

	setLastResult(m_Classifier.toString(), detail.toString(), validation.toSummaryString());

	return lastResult;
}

/**
 * main�� �׽�Ʈ�� (GP �ԷµǾ� ����)
 */
/**
 * @param args
 * @throws Exception
 */

public static void main(String[] args) throws Exception {
	WekaAPI w = new WekaAPI();
	WekaAPI e = new WekaAPI();

	//function과 option 설정
	String function = "DL4J";


	String option = "-S 1 -iterator \"weka.dl4j.iterators.DefaultInstancesIterator -bs 1\" -layers " +
			"\"weka.dl4j.layers.DenseLayer -nOut 200 -activation tanh -adamMeanDecay 0.9 -adamVarDecay 0.999 -biasInit 1.0 -biasL1 0.0 -biasL2 0.0 -blr 0.001 -dist " +
			"\\\"weka.dl4j.distribution.NormalDistribution -mean 0.001 -std 1.0\\\"" +
			" -dropout 0.0 -epsilon 1.0E-6 -gradientNormalization None -gradNormThreshold 1.0 -L1 0.0 -L2 0.0 -name " +
			"\\\"Hidden layer\\\" -lr 0.001 -momentum 0.9 -rho 0.0 -rmsDecay 0.95 -updater NESTEROVS -weightInit XAVIER\" -layers " +
			"\"weka.dl4j.layers.DenseLayer -nOut 200 -activation tanh -adamMeanDecay 0.9 -adamVarDecay 0.999 -biasInit 1.0 -biasL1 0.0 -biasL2 0.0 -blr 0.001 -dist " +
			"\\\"weka.dl4j.distribution.NormalDistribution -mean 0.001 -std 1.0\\\"" +
			" -dropout 0.0 -epsilon 1.0E-6 -gradientNormalization None -gradNormThreshold 1.0 -L1 0.0 -L2 0.0 -name " +
			"\\\"Hidden layer\\\" -lr 0.001 -momentum 0.9 -rho 0.0 -rmsDecay 0.95 -updater NESTEROVS -weightInit XAVIER\" -layers " +
			"\"weka.dl4j.layers.OutputLayer -activation identity -adamMeanDecay 0.9 -adamVarDecay 0.999 -biasInit 1.0 -biasL1 0.0 -biasL2 0.0 -blr 0.001 -dist " +
			"\\\"weka.dl4j.distribution.NormalDistribution -mean 0.001 -std 1.0\\\" -dropout 0.0 -epsilon 1.0E-6 -gradientNormalization None -gradNormThreshold 1.0 -L1 0.0 -L2 0.0 -name " +
			"\\\"Output layer\\\" -lr 0.001 -lossFn LossMSE() -momentum 0.9 -rho 0.0 -rmsDecay 0.95 -updater NESTEROVS -weightInit XAVIER\"" +
			" -logFile \"C:\\Program Files\\Weka-3-8\" -numEpochs 20 -algorithm STOCHASTIC_GRADIENT_DESCENT";
	//String option =  "-S 1 -iterator \"weka.dl4j.iterators.DefaultInstancesIterator -bs 1\" -layers \"weka.dl4j.layers.OutputLayer -activation identity -adamMeanDecay 0.9 -adamVarDecay 0.999 -biasInit 1.0 -biasL1 0.0 -biasL2 0.0 -blr 1.0E-6 -dist \\\"weka.dl4j.distribution.NormalDistribution -mean 0.001 -std 1.0\\\" -dropout 0.0 -epsilon 1.0E-6 -gradientNormalization None -gradNormThreshold 1.0 -L1 0.0 -L2 0.0 -name \\\"Output layer\\\" -lr 1.0E-6 -lossFn LossMCXENT() -momentum 0.9 -rho 0.0 -rmsDecay 0.95 -updater NESTEROVS -weightInit XAVIER\" -logFile \"C:\\Program Files\\Weka-3-8\" -numEpochs 10 -algorithm STOCHASTIC_GRADIENT_DESCENT";
	//String option = "-C 5.0 -N 1 -I \"weka.classifiers.functions.supportVector.RegSMOImproved -T 0.001 -V -P 1.0E-12 -L 0.001 -W 1\" -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -L -C 250007\"";
	String[] defaultname={"1","5","6","7"};
	String[] crossname={"1","5","6","7"};

	//attr 설정
	String attr="4";
	int indexNum=Integer.parseInt(attr);

	WritableCellFormat floatformat = new WritableCellFormat(NumberFormats.FLOAT);
	NumberFormat fiveformat = new NumberFormat("#.#####");
	WritableCellFormat userformat = new WritableCellFormat(fiveformat);
	//file명 설정
	File file = new File("C:\\Users\\찬재\\Desktop\\"+function+"cs"+".xls");
	workbook = Workbook.createWorkbook(file);

	for (String num : crossname){
	//데이터 설정 _relTrain _less
	/*String dataFile_y = "resource/data/attr"+attr+"/default/case0"+num+"/lat"+attr+"_relTrain_case_0"+num+".arff";
	String dataFile_x = "resource/data/attr"+attr+"/default/case0"+num+"/lon"+attr+"_relTrain_case_0"+num+".arff";*/
	String dataFile_y = "resource/data/attr"+attr+"/cross/case0"+num+"/lat"+attr+"_less_case_0"+num+".arff";
	String dataFile_x = "resource/data/attr"+attr+"/cross/case0"+num+"/lon"+attr+"_less_case_0"+num+".arff";
	int fold = 10;
	int seed = 1;
	//name 설정
	String name=function+" standardized "+num;
	w.buildModel(function, option, dataFile_y,indexNum);
	e.buildModel(function, option, dataFile_x,indexNum);
	//String[] result = w.runCrossValidation(fold, seed);

	//데이터 설정 _relTest
	/*String[] result_y = w.runTestSet("resource/data/attr"+attr+"/default/case0"+num+"/lat"+attr+"_relTest_case_0"+num+".arff",indexNum);
	String[] result_x = e.runTestSet("resource/data/attr"+attr+"/default/case0"+num+"/lon"+attr+"_relTest_case_0"+num+".arff",indexNum);*/
	String[] result_y = w.runTestSet("resource/data/attr"+attr+"/cross/case0"+num+"/lat"+attr+"_case_0"+num+".arff",indexNum);
	String[] result_x = e.runTestSet("resource/data/attr"+attr+"/cross/case0"+num+"/lon"+attr+"_case_0"+num+".arff",indexNum);

	System.out.println(result_y[0]);
	System.out.println(result_y[1]);
	System.out.println(result_y[2]);
	System.out.println(result_x[0]);
	System.out.println(result_x[1]);
	System.out.println(result_x[2]);

	// result_y에서 y좌표 추출
	Float[][] val_y =FormatData(result_y);
	Float[][] val_x =FormatData(result_x);


	Float[] Var_x=new Float[val_x[0].length];
	Float[] Var_y=new Float[val_x[0].length];

	for(int i=0;i<val_x[0].length;i++)
	{
		Var_x[i]=val_x[0][i]-val_x[1][i];
		Var_y[i]=val_y[0][i]-val_y[1][i];
		Var_x[i]=Math.abs(Var_x[i]);
		Var_y[i]=Math.abs(Var_y[i]);
	}
	//(x+y)/2
	Float[] MAE=new Float[val_x[0].length];
	MAE[0]=(float)0;
	for(int i=1;i<val_x[0].length;i++)
	{
		//MAE[i]=(Var_x[i]+Var_y[i])/2+MAE[i-1];
		MAE[i]=(float) Math.sqrt(Math.pow(Var_x[i],2)+Math.pow(Var_y[i],2))+MAE[i-1];
	}
	System.out.println(MAE[val_x[0].length-1]/(val_x[0].length));
	Float MAEresult=MAE[val_x[0].length-1]/(val_x[0].length);
	GenerateExcel(file,val_x[0],val_x[1],val_y[0],val_y[1],Var_x,Var_y,MAEresult,name,num);

	}
	workbook.write();
	workbook.close();
}

public static Float[][] FormatData(String[] result) {
	// TODO Auto-generated method stub
	String[] result_1=result[1].split("\n");

	/*System.out.println(Arrays.toString(result_1));*/
	String[][] result_2=new String[result_1.length][];
	for(int i=0;i<result_1.length;i++)
		result_2[i]=result_1[i].split("\t");
	Float[][] val = new Float[2][];
	for(int i=0;i<2;i++)
		val[i] = new Float[result_1.length+1];
	for(int i=0;i<result_1.length+1;i++)
	{
		if(i==0)
			val[0][i]=(float) 0;
		else
			val[0][i]=Float.parseFloat(result_2[i-1][1])+val[0][i-1];
	}
	for(int i=0;i<result_1.length+1;i++)
	{
		if(i==0)
			val[1][i]=(float) 0;
		else
			val[1][i]=Float.parseFloat(result_2[i-1][2])+val[1][i-1];
	}
	return val;
}

public static void GenerateExcel(File file,Float[] real_x,Float[] pre_x,Float[] real_y,Float[] pre_y,Float[] Var_x,Float[] Var_y,Float MAEresult,String name,String num)
{
  /*  // 테스트 데이터
    HashMap hm_0 = new HashMap() ;
    hm_0.put("name", "홍길동") ;
    hm_0.put("age", "21") ;
    HashMap hm_1 = new HashMap() ;
    hm_1.put("name", "김영희") ;
    hm_1.put("age", "20") ;

    List list = new ArrayList();
    list.add(hm_0) ;
    list.add(hm_1) ;
     */

    try{

        // 파일 생성


        // 시트 생성
        workbook.createSheet(name, sheetnum);
        sheet = workbook.getSheet(name);
        sheetnum++;
        // 셀에 쓰기
        label = new Label(0, 0, "real_x");
        sheet.addCell(label);

        label = new Label(1, 0, "pre_x");
        sheet.addCell(label);

        label = new Label(3, 0, "real_y");
        sheet.addCell(label);

        label = new Label(4, 0, "pre_x");
        sheet.addCell(label);

        label = new Label(6, 0, "Var_x");
        sheet.addCell(label);

        label = new Label(7, 0, "Var_y");
        sheet.addCell(label);


        label = new Label(9, 0, "MAE");
        sheet.addCell(label);





      // 데이터 삽입
        for(int i=0; i < real_x.length; i++){


        	number= new jxl.write.Number(0,(i+2),real_x[i]);
            sheet.addCell(number);

            number= new jxl.write.Number(1,(i+2),pre_x[i]);
            sheet.addCell(number);

            number= new jxl.write.Number(3,(i+2),real_y[i]);
            sheet.addCell(number);

            number= new jxl.write.Number(4,(i+2),pre_y[i]);
            sheet.addCell(number);

            number= new jxl.write.Number(6,(i+2),Var_x[i]);
            sheet.addCell(number);

            number= new jxl.write.Number(7,(i+2),Var_y[i]);
            sheet.addCell(number);


        }
        number= new jxl.write.Number(9,2,MAEresult);
        sheet.addCell(number);






    }catch(Exception e){
        e.printStackTrace();
    }
}

/**
 * ��� ������� ����
 * @param path		���� ��ġ
 * @return
 */

public boolean saveModel(String path) {
	if(!isBuild())
		return false;

	File sFile = new File(path);

	OutputStream os;
	try {
		os = new FileOutputStream(sFile);
		if (sFile.getName().endsWith(".gz")) {
		os = new GZIPOutputStream(os);
	}
	ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
	objectOutputStream.writeObject(m_Classifier);
	m_Training = m_Training.stringFreeStructure();
	if (m_Training != null) {
		objectOutputStream.writeObject(m_Training);
	}
	objectOutputStream.flush();
	objectOutputStream.close();

	return true;
} catch (Exception e) {
	// TODO Auto-generated catch block
			e.printStackTrace();

			return false;
		}
	}
}
