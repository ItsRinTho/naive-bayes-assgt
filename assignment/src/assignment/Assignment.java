package assignment;

import java.io.File;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Assignment {

    public static void main(String[] args) throws IOException, Exception {
       
        Instances data = getData();
        
        NaiveBayes naiveBayes = new NaiveBayes();
        Evaluation eval = new Evaluation(data);
        
        naiveBayes.buildClassifier(data);
        eval.crossValidateModel(naiveBayes, data, 10, new Random(1));
        
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
        
    }

    private static Instances getData() throws IOException, Exception {
        final String DATASETS = "src\\datasets";
        File dir = new File(DATASETS);
        
        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(dir);
        
        Instances instance = loader.getDataSet();
        
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(instance);
        Instances dataFiltered = Filter.useFilter(instance, filter);

        if(dataFiltered.classIndex() == -1) {
            dataFiltered.setClassIndex(dataFiltered.numAttributes() - 1);
        }
        return dataFiltered;
    }
}
