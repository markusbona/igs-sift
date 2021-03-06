/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.neuroph.util.norm;

import org.neuroph.core.learning.TrainingElement;
import org.neuroph.core.learning.TrainingSet;

/**
 * Max normalization method, which normalize data in regard to max element in training set (by columns)
 * Normalization is done according to formula:
 * normalizedVector[i] = vector[i] / max[i]
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class MaxNormalizer implements Normalizer {
    double[] max; // contains max values for all columns
    
    
    @Override
   public void normalize(TrainingSet<? extends TrainingElement> trainingSet) {

        findMaxVector(trainingSet);

        for (TrainingElement trainingElement : trainingSet.elements()) {
            double[] input = trainingElement.getInput();
            double[] normalizedInput = normalizeMax(input);
            trainingElement.setInput(normalizedInput);
        }

    }    
   
    private void findMaxVector(TrainingSet<? extends TrainingElement> trainingSet) {
        int inputSize = trainingSet.getInputSize();
        max = new double[inputSize];

        for (TrainingElement te : trainingSet.elements()) {
            double[] input = te.getInput();
            for (int i = 0; i < inputSize; i++) {
                if (Math.abs(input[i]) > max[i]) {
                    max[i] = Math.abs(input[i]);
                }
            }
        }        
    }   
    
    public double[] normalizeMax(double[] vector) {
        double[] normalizedVector = new double[vector.length];
        
        for(int i = 0; i < vector.length; i++) {
            if (max[i]>1)
                normalizedVector[i] = vector[i] / max[i];
        }
        
        return normalizedVector;
    }     
   
}
