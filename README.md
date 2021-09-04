# License-Plate-Detection
License plate detection using different networks

## I. Introduction
Machine Learning and Artificial Intelligence is very important to this world today, every single technological advances has to do with some machine learning and artificial intelligence including computer vision, natural language processing, autonomus driving, and much more. In this project, I wanted to test computer vision, a part of computer science and improve on it. I choose a very popular concept, license plate detection to carry out my research. First of all, my mentor and I found avery well organized article on computer vision and license plate detection. Then I used this as a framework for my research. 

## II. Research
In order to edit the code, I needed to do more research including looking into the Keras applications and documentation as this was used in the framework. I have also dived deeply in to the reference code and figured out what individual steps are doing.

## III. Project
In order to run the project, I split it into three different sections. The first section is let the computer to be able to detect where the license plate is given a picture. The second step is to recognize where the individual letters are in the lincense plate and then clean them in order to make the computer easier to detect. The last and the most important step is to train the computer using different networks and then detect the letters using machine learning.

### &nbsp;&nbsp;&nbsp; Part 1: Extracting License Plate Location from Image
&nbsp;&nbsp;&nbsp; For this part, 

### &nbsp;&nbsp;&nbsp; Part 2: Recognizing Letters within the License Plate
&nbsp;&nbsp;&nbsp; In this part of the project I had to first greyscale and turn the license plate into a easier formate to use. 

### &nbsp;&nbsp;&nbsp; Part 3: Training the Computer with different Networks and using them to detect the letters.
&nbsp;&nbsp;&nbsp; Training the computer takes a long time and memory. I first researched what types of networks that I could use with the frameword and I found out that I could use MobileNet, ResNet, and Xception.

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MobileNet:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This is the network that the framework came with. Even though there was a lot of errors and it took me a long time to get it running, I was able to produce some of the results that the original author had. MobileNet uses lightweight deep convolutional neural networks to provide an efficient model for mobile and embedded vision applications. 

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ResNet:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ResNet worked flawlessly within the network but it did took a lot longer than MobileNet on Google Colab. ResNet uses residual learning and builds pyramidal cells so that layers could be skipped. 

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Xception:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Xception took around the same time as ResNet. Xception is a deeper implementation of Inception which uses Deapthwise Separable Convolutions to train the data. 

### &nbsp;&nbsp;&nbsp; Challenges:
&nbsp;&nbsp;&nbsp; Throughout this whole project, many problems was encountered. First of all, I discovered that Google Colab was really slow in running the models and tests so I decided to use my own computer with the gpu to hopefully make the training go faster. Then I discovered that even with my gpu, the computer still does not run as fast as Google Colab and my memory was running out. In the end, I switched back to Google Colab and tested the rest of the project there. When using the different networks, problems also occured when chosing which version of the librariesare used such as for the keras application and tensorflow. Throughout the process, there was many bugs as well that was discovered. I was able to overcome all the challenges and tested and experienced with machine learning in deapth. 

### &nbsp;&nbsp;&nbsp; Results:


## Conclusion:


## References
