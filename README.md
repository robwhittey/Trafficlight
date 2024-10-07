# Trafficlight

<h2>§1 - Introduction</h2>

Since the dawn of the first driverless car in 1939 (controlled by electromagnetic fields), technology through the years has advanced exponentially and the reality of a driverless car being sold commercially is closer than ever. With so many different variables to consider from other vehicle positioning to obstacle avoidance, there is one sector in particular that we will be looking into. The recognition and classification of road signs. 

In this report, we will be looking at optimising images for better model performance, seeing what size images, how many and what colour channel is best. Also, we will be experimenting with two convolutional neural networks; one pretrained and one custom made. The goal is to produce the best model that is above 96% accurate and with the lowest loss. 
<h2>§2 – Overview of the data</h2>

The dataset provided contains 877 images of four classes to predict. There are 652 speed limit signs, 88 crosswalk signs, 76 stop signs and 61 traffic lights. The images are in a variety of conditions with some in different lighting, taken from obscure angles and with the occasional image having a removed background. Since the goal is to classify these images with a deep learning model which will later be used for classification for driverless cars, we will not be removing any of the images. 

All the images vary in width and height, and fortunately, the annotations provided contain coordinates for the bounding box. From here, we can go one of two ways. We can either reshape all the images to one size or crop the images to their bounding boxes and reshape from there. Since the annotation type is a string, we must first label encode the classes for the deep learning model to process. 
<h2>§3 – Image pre-processing</h2>

As mentioned in §2-Overview of the data, there is not an equal amount of each class for prediction with nearly 75% of the data being speed limits. This is known as an unbalanced training data with an effect of unoptimized learning and failure to unbiasedly validate (Chatterjee. S, 2018). In order to give the models the best chance for accurate prediction (especially with the validation dataset), we must produce more images of crosswalks, stop signs and traffic lights. To do this we will apply image augmentation; applying different manipulation techniques to an image to produce more. These could include and not limited to cropping, rotation, and mirroring. Since the aim is to have an equal dataset of each class type, we must produce enough images of the minimum class to be close to the maximum class. Therefore, we will augment the three minimum classed images 10 times. Alternatively, one could gather more images to add to the training data but, I believe it would be best to use the signs of the particular country (which is not known) and augmentation is a simple approach. 

The images are also found to be in different lighting conditions. Since our goal is to train driverless cars which will be used in all conditions, it will be beneficial to have un-altered images with poor lighting for the model to train with. Therefore, we will not be artificially increasing light exposure or other means for these images as that would require the driverless car to do so and would take too much processing time.

To ensure our pre-processed images are ready to train our models and produce the best success rate, we will conduct a few quick tests on training the VGG16 model (see §4 – Model VGG16). Firstly, as a control, we will train the model using the natural images with no augmentation and scaled to 224x224. This will give us a baseline to ensure any changes we do to the global images improve on the accuracy (we will look into accuracy and loss in more detail later). We will then introduce the augmented images, resizing the images again to 50x50, and greyscale. The most successful combination of pre-processing of the images will then go forward for training and analysis.

Figure 1: Graphs of accuracy and loss over 10 epochs of training and validation with 
VGG16 and natural images. 

The above graph (figure 1) shows our baseline for referencing improvement. As we can see, during training, the accuracy increases with time and the loss is reducing. But, our problem lies with the validation dataset which does not appear to follow the trend of training. Instead, the loss hovers around the 1.0 mark and the accuracy around 75%; which is not great for sign recognition. 

Next, we will look at introducing augmented images to have a balanced dataset. The images are still 224x224 and the model has had no alterations. In the below graphs (Figure 2), we can see a drastic improvement to both the learning and loss in training and validation. Over the 10 epochs, the accuracy and loss of validation follow the trends found in training. 

Figure 2: Graphs of accuracy and loss over 10 epochs of training and validation with 
VGG16 and augmented images

Now we will look at using 50x50 images instead of 224x224. We rescale the images as well as their augmented counterparts to see if there are any improvements. We find that these images do not take as long for the model to learn since the dimension is vastly decreased. Although, what we do lose is crucial detail in the imagery and notice from the graphs below (figure 3), a much shallower gradient of learning. Since our aim original aim was to utilise the best pre-processing method that has better results after 10 epochs in training, we will not use rescaled images of 50x50. 


Figure 3: Graphs of accuracy and loss over 10 epochs of training and validation with 
VGG16 and augmented images rescaled to 50x50

Finally, we will look at grayscale images. Unfortunately, VGG16 along with imagenet weights do not accept single channel grayscale images. To avoid this issue, and keeping the test fair, we will still use the images mapped to grayscale but with 3 channels. This is not ideal as we will still have the same dimensionality as RGB colour images and the whole point of reducing to grayscale is to reduce the number of channels. Fortunately, the result should be equivalent but will take more time. From the graphs below (figure 4), we can see that the model does indeed learn with the training accuracy increasing and the training loss decreasing but, the validation dataset does not follow suit. This is therefore not a good pre-processing technique.


Figure 4: Graphs of accuracy and loss over 10 epochs of training and validation with 
VGG16 and augmented grayscale images 

From the tests conducted above, we can therefore conclude that our best course of action for giving our models the best chance at a higher success is with a balanced dataset scaled to 224x224 in RGB. 


<h2>§4 – Model VGG16 </h2>

The award winning VGG16 outperformed other models in the 2014 ImageNet challenge. Trained with 1000 image classes, it had an overall accuracy of 92.7% (Simonyan, K. and Zisserman, A. 2015). Since the model was trained for image recognition and detection, it is therefore a good option as a pretrained model for us to work with. After downloading the original weights for the model, all is needed are an extra couple of trainable layers for the model to work with. 

<h3>§4.1 – Model Architecture</h3>

The image below shows the architecture for the VGG16 model. It has a depth of 23 layers and an input shape of (224, 224, 3). The original output of the model takes a shape of 1000 (since it was originally used for images belonging to 1000 different classes). 

Figure 5: VGG16 model architecture
To adapt the model for our benefit, we must remove the final layer of the VGG16 model and add some addition layers. More importantly, the output layer must be 4 since we only have 4 classes of images. 


Figure 6: Untrainable VGG16 model with additional trainable layers 




<h3>§4.2 - Model fine tuning and performance</h3>

When it comes to fine tuning a model, there are many alterations that can be done. What we can change in a model are called hyperparameters and there are a few hyperparameters that we can alter. Examples of these are amount of layers, activation functions, batch size, epochs, and learning rate. In this section, we will be changing each of them to obtain the optimum result for this model. 

To begin with, we will alter the number of epochs. Epochs are defined as the number of times we send the data back and forth through the model for learning. In our previous test (see §3 – Image pre-processing), we set the epoch level to 10 which gave us a good indication of whether changing the data was improving learning. Now that we have optimal data, we will see how the model learns after 50 epochs. From the graphs below (see figure 7), we can see a great improvement in both accuracy and loss of both training and validation up to 10 epochs. Thereafter, the learning is gradual with virtually no change. The following are results from each epoch:

Epoch 10/50

17/17 [==============================] - 57s 3s/step - loss: 0.0124 - acc: 0.9995 -   val_loss: 0.1160 - val_acc: 0.9604

...

Epoch 49/50

17/17 [==============================] - 55s 3s/step - loss: 3.1796e-05 - acc: 1.0000 - val_loss: 0.1556 - val_acc: 0.9615

Epoch 50/50

17/17 [==============================] - 55s 3s/step - loss: 0.1727 - acc: 0.9680 -   val_loss: 0.2443 - val_acc: 0.9297

Unfortunately, the last epoch experienced a setback in progress but comparing the second to last with the tenth epoch, we see that in 40 epochs, the validation loss had increased and the accuracy reduced by only 0.03. 

Figure 7: VGG16 after 50 epochs

Next, we will look at reducing and increasing the batch size. This is the number of images sent through the model each time. For this, the original batch size was 128. Here, we will do a single test up and down of this number and the batch sizes we will use are 64 and 256. The figures below (see figure 8 and 9) show the results for the different batch sizes. We notice immediately that the smaller batch size of 64 does not perform well when it comes to validation for both accuracy and loss. At first glance, a batch size of 256 seams promising but comparing with the original batch size of 128 (see Figure 2), we notice that the original has a lower tendency for loss which for this test is better. 


Figure 8 (Left): VGG16. Batch size 64.         Figure 9 (Right): VGG16. Batch size 256

The learning rate of a model indicates the incrementation taken for learning to regulate the weights. A larger learning rate covers more area of minimising the loss and a smaller learning rate takes longer. The issue with a smaller learning rate is the risk of being caught in a local minimum whereas a larger learning rate may pass over the local and global minima altogether. For this, we will look at the model evaluation as well as the breakdown per epoch. As we can see form the figures below, the smaller learning rate (figure 10) is much smoother compared to the original (figure 2) and the greater learning rate (figure 11). But, the smaller learning rate of 1e-5 does not reach the same level as original learning rate of 1e-4. However, the reliability of a smoother curve makes this particular value of a learning rate more appealing to use, especially in conjunction with more epochs.


Figure 10 (Left): VGG16. Lr = 1e-5           Figure 11 (Right): VGG16. Lr = 1e-3

Now, we will look at using different activation functions for the hidden layer. Currently, we are using RELU which is computationally efficient. We will try other activation functions such as sigmoid, tanh, and softmax and compare the results. From the graphs below (see figure 12), it is clear that softmax is not a good activation function for the hidden layer since the loss function is not minimised within the 10 epochs. In the model evaluation, tanh was the best with a loss of 0.005 and accuracy of 1.000. 


Figure 12: VGG16. Tanh, sigmoid, and softmax activation functions (From left to right).

Finally, we will look at combining the best of each test and see if the performance is improved in terms of accuracy, especially for the validation dataset. We will use a learning rate of 1e-5 with a greater amount of epochs (15) and tanh as the activation function. Looking at the last epoch and the evaluation of the model, we notice that there is not much of an improvement compared to earlier tests. They are as followed:

Epoch 15/15

17/17 [==============================] - 54s 3s/step - loss: 0.0545 - acc: 0.9939 – val_loss: 0.1906 - val_acc: 0.9407

Evaluation:

67/67 [==============================] - 35s 517ms/step - loss: 0.0455 - acc: 0.9986


Figure 13: VGG16 with tanh activation, lr = 1e-5 and 15 epochs. 
<h2>§5 – Custom model</h2>

In this section, we will look at constructing our own architecture of a model and adapting it in various ways to produce the best model. We will begin with a basic model and may find we need to add layers, change activation functions and other hyperparameters.








<h3>§5.1 – Model Architecture</h3>

Below (figure 14) is the model architecture we will start with. It is based on the ideas found in VGG16, VGG19 and AlexNet models but more simplified.

Figure 14: Starting Architecture for custom model.

The Conv2D layers will use 12 layers and relu as an activation function and the dense_17 and dense_18 layers will also have relu activation.

<h3>§5.2 - Model fine tuning and performance</h3>

As the model stands, with a learning rate of 1e-4, after 10 epochs our model evaluation is as so:

	Epoch 10/10

17/17 [==============================] - 5s 282ms/step - loss: 1.0845 - acc: 0.5195 - val_loss: 1.0644 - val_acc: 0.5549

Evaluation:

67/67 [==============================] - 2s 26ms/step - loss: 1.0466 - acc: 0.5577

These results are far from ideal and far from what we were finding with the pretrained model in §4 – VGG16 Model. This is our first example of where pretrained models are both faster at learning and more efficient methods for quick learning. 

To continue, it is evident that we will need to have a large amount of epochs in order to successfully train the custom model. But, to give our model the best chance of success, like before, we will stick with testing up to 10 epochs for various combinations of hyperparameter manipulation and continuing with higher epochs once we find the optimum design.

Figure 15: Custom model evaluation per epoch (best)

After changing each hyperparameter and plotting the result, the best graph (see figure 15 above) shows a trend for both train and validation for accuracy and loss. The optimum combination found also has the best accuracy and loss after 10 epochs as follows:

Epoch 10/10

17/17 [==============================] - 6s 332ms/step - loss: 0.7340 - acc: 0.7051 - val_loss: 0.7344 - val_acc: 0.6978

67/67 [==============================] - 3s 39ms/step - loss: 0.6755 - acc: 0.7376

As with before, the presence of a tanh activation functions improved the performance of the model. With greater depth, the additional Conv2D layers worsened the accuracy and raising and lowering the number of nodes in the dense layer also had a negative effect. We found that increasing the learning rate to 5e-4 had a smoother and better success rate. Finally, adding more filters to the conv2D layers added more noise to the graph (see code).

With the optimum solution found, it is now time to increase the epoch level to 150. An issue we are likely to face when doing this is overfitting the model to the training data. As we can see from the graph below (figure 16), we do find this to be the case. Both the validation accuracy and loss trail off on their own course after 10 epochs.


Figure 16: optimum solution after 150 epochs

Taking a different approach, completely disregarding what we originally considered optimal, I then went forward and tried a more dense model similar to the custom model with more conv2D layers and more filters. The below graphs illustrate what was produced (figure 17).


Figure 17: custom model after 150 epochs with more layers and more filters.

As we can see, a similar issue is had that we found in the optimal solution but the trend of the validations are closer to that of the training. Unfortunately, there is a lot of noise, and this is due to overfitting. 

<h3>Summary</h3>

Despite not being able to produce a custom model that is superior to the pretrained model, we were still able to classify images to an accuracy of above 96%. We were able to minimise the loss to below 8% in the custom model too. But the issue lies with the validation dataset. Perhaps with a different pre-processing approach would solve this or it is solely down to the models and not finding the true optimal solution.

<h2>Conclusion</h2>

In conclusion, the VGG16 as a pretrained model is a perfect platform model to build off and allows the user to have a high accuracy with lower number of epochs. Unfortunately, with both models, we experience issues with the validation dataset. Perhaps with different metrics we would be able to minimise this issue. But there are still plenty of other hyperparameters that could have been changed to alter the performance. The dataset was unbalanced and perhaps with more images being added as opposed to augmentation would yield a better result. Next time, I would like to try cropping the images to their bounding box for training and see how this affects the validation accuracy and loss. 






















<h2>Bibliography</h2>

Baheti, P. (2022). 12 Types of Neural Networks Activation Functions: How to Choose? [online] www.v7labs.com. Available at: https://www.v7labs.com/blog/neural-networks-activation-functions. 
Guo, T., Dong, J., Li, H. and Gao, Y. (2017). Simple convolutional neural network on image classification. 2017 IEEE 2nd International Conference on Big Data Analysis (ICBDA)(. doi:10.1109/icbda.2017.8078730.
He, K., Zhang, X., Ren, S. and Sun, J. (2015). Deep Residual Learning for Image Recognition. [online] Available at: https://arxiv.org/pdf/1512.03385.pdf. 
Brownlee, J. (2019). Architectural Innovations in Convolutional Neural Networks for Image Classification. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/review-of-architectural-innovations-for-convolutional-neural-networks-for-image-classification/  [Accessed 19 Dec 2022].
Learning, G. (2021). Everything you need to know about VGG16. [online] Medium. Available at: https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918. 
Rocco, I., Arandjelovic, R. and Sivic, J. (2019). Convolutional Neural Network Architecture for Geometric Matching. IEEE Transactions on Pattern Analysis and Machine Intelligence, [online] 41(11), pp.2553–2567. doi:10.1109/tpami.2018.2865351.
Shubrashankh Chatterjee (2018). Deep learning unbalanced training data?Solve it like this. [online] Medium. Available at: https://towardsdatascience.com/deep-learning-unbalanced-training-data-solve-it-like-this-6c528e9efea6. 
Simonyan, K. and Zisserman, A. (2015). Published as a conference paper at ICLR 2015 VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION. [online] Available at: https://arxiv.org/pdf/1409.1556.pdf. 
