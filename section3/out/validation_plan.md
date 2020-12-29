# Validation plan

## Intended use (Purpose)

The goal of our AI system is to support radiologist with measuring the hippocampus volume of patients based on their 
brain MRI scan. This may be helpful in tracking the progression of certain brain disorders like Alzheimer.

The algorithm is supposed to be integrated after the MRI scanner sends the volume to PACS system. Here, the correct 
study is selected (which one is the correct one?) followed by the application of the cropping algorithm. 
Subsequently, the model is applied to segment the volume, calculate the 
hippocampus volume and create a report, which is stored again in the PACS system. Finally, the radiologist can view this
report in the OHIF viewer. (TODO specify a bit clearer after simulation exercise)

## Validation cohort

Unfortunately, we don't have any information about the demographics and comorbid diseases in the training data set. 
Therefore, the validation data set should come from a typical clinical setting containing a variety of different age 
groups and other potential diseases in order to measure the performance and suitability of the model in these 
different patient segments. Although, as the specified purpose is mainly for tracking Alzheimer progress, the majority of 
studies should be taken from Alzheimer patients, ideally multiple volumes per patient in different stages.

## Ground truth 

The ground truth for the validation data set should be created by a group of experienced radiologist, for example 5. 
Each of them would segment all volumes to the best of their knowledge. Based on these single segmentations we could 
create a ground truth by marking all voxels as hippocampus, which were marked by the majority of radiologists 
(for example at least 3 out of 5).    

## Performance metrics

In order to evaluate how well our model performs in comparison with the created ground truth, we could use Dice and 
Jaccard coefficients. Here, we would compare the average score of our model with the scores of the single radiologist 
regarding their segmentations and the ground truth. If our model does not perform significantly worse than the average
radiologist, this would be a good indication for its usefulness for tracking Alzheimer progress. This analysis should
be repeated for different patient segments (age, gender, other diseases) in order to determine the applicability of the 
algorithms for these segments.
   
Besides, its absolute accuracy, we could also measure its relative accuracy. More precisely, even if the model is not 
perfectly accurate in determine the exact volume, it might be sufficient for the task if its good enough to determine 
the change in subsequent studies. For this purpose we would look at the fractional volume change per patient and 
subsequent studies and compare the ground truth and the model outputs. Even if the model would systematically 
underestimate the volume, it might still perform well on this second metric and hence be useful for the tracking.       
