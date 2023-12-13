# Gradient_CNN

The code in this repository is used to generate controlled volume fractions by updating the flow rates of syringe pumps based on image feedback analyzed by CNNs. The 'Img_loader_2_classes.py' loads training, testing and validation images present in separate, pre-made folders. The 'CNN_model_120_x_120_10x' defines the CNN architecture and the 'Train_CNN.py' function uses 'CNN_model' and 'Img_loader_2_classes' to train the shallow CNN and generate a model. The 'pumps_function.py' sets syringe diameters, units for flow rates. This function requires the QmixSDK library to operate the syringe pumps. The main function 'AI_gradient_CNN.py' uses the trained model evaluations to alternate betweeen a low and high volume fractions, including overshooting beyond the trained classes. 


Data analysis: An example MATLAB code 'Extract_spheroids.m' that uses the YOLOv4 spheroid and fluorescent beads detections is given. The code extracts spheroid area and bead fluorescence and produces a scatter plot.
