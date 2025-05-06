* folder structure 

This contains the following folders
1) data -> This contains the datatset and the code to download the dataset

2) exploreDataset -> This folder contains codes for visualization of dataset
    -> Load data using waveform DB library and explored the .hea , .dat and .atr files
    -> Use plots to visualize QRS complexes

3) models -> This contains evaluating images of the model, scripts for testing the model (used only for my clearence )
             and the scripts for models

             for further clarity -> CnnModel.py contains the CNN model
                                    Evaluate.py contains the evaluating metrics 
                                    MainPipeLine.py contains the main code to develop the model (Consider all records)
                                    MainTrain.py contanis code to train the model for all records 
                                    RunOneRecord.py contains the pipeline to develop model for one record 
                                    TrainOneRecord.py contains code to train the model for one record 

4) preProcessing -> this contains the pre0processing steps of ECG signal 
    -> Denoising 
	    -> Used a band pass filter to remove frequencies out side the tyoical ECG range
	    -> Used a notch filter to remove power line interference
	    -> Removes baseline wander using a moving average approach
	    -> Returns the denoised signal

    -> Segmenting
	    -> Locate R peaks using neurokit2 library and defined the size of QRS complex as 250 samples(0.694 seconds)
	    -> Returns an array of segments and an array of R peaks located by neurokit2 library

    -> Normalization 
	    -> Returns an array of normalized beats

    -> Create labels
	    -> Create labels for the relevant R peak, related annotations in the .atr file
	    -> N, L, R considered as Normal(0) and others as abnormal(1)
	    -> Returns an array of labels related to the segments

    -> Class balancing 
	    -> Uses SMOTE algorithm for class balancing
	    -> Returns balanced set of samples

To run the model 
    -> cd yourpath/ecgClassificationCnnModel
    -> python -m models.MainPipeLine