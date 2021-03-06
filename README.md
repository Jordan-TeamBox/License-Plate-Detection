# License-Plate-Detection
Artificial Intelligence and Machine learning has become an inseparable technology in this world. Many of our equipment require these technologies to function. We are going to see how one part of machine learning, licence plate detection, is implemented and test out the different neural networks that could be used to train the data.

## I. Introduction
Machine Learning and Artificial Intelligence is very important to this world today, every single technological advances has to do with some machine learning and artificial intelligence including computer vision, natural language processing, autonomus driving, and much more. This project tests computer vision, a part of computer science, and improve on it. License Plate detection is being researched. This research uses Quang Nguyen's medium page (Detect and Recognize Vehicle’s License Plate with Machine Learning and Python) as a framework. 

## II. Research
In order to edit the code and test the program, extensive research was conducted, this includes looking into the Keras applications and documentation as this was used in the framework. Also, diving deeply in to the reference code is very helpful to figured out what the individual steps are doing.

## III. Project
The project is done in three different sections. The first section is letting the computer detect where the license plate is given a picture. The second step is to recognize where the individual letters are in the lincense plate and then clean them in order to make the computer easier to detect. The last and the most important step is to train the computer using different networks and then detect the letters using machine learning.

### &nbsp;&nbsp;&nbsp; Part 1: Extracting License Plate Location from Image
&nbsp;&nbsp;&nbsp; For this part, a pretrained model using wpod net is used and the following function uses wpod to detect the license plate location with its respective coordinates. [1]

``` Python
def get_plate(image_path, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor
```
The result is

<img src="Images/part1_result (1).jpg">

### &nbsp;&nbsp;&nbsp; Part 2: Recognizing Letters within the License Plate
&nbsp;&nbsp;&nbsp; In this part of the project, many steps has to be done. In order to process the images, this function makes sure the pictures are equal sizes. [2]

``` Python
def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img
```

Then other techniques are used to get what we want, such as converting the images to 255 scale, then grayscale, then blurring it and also setting a threshhold value to change it to binary and finally, dilate the image. Which is achieved using the following lines. [2]

``` Python
if (len(LpImg)): #check if there is at least one license image
      # Scales, calculates absolute values, and converts the result to 8-bit.
      plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    # convert to grayscale and blur the image
      gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
      binary = cv2.threshold(blur, 180, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
      kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
      thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
```
The result looks like this

<img src="/Images/threshding (1).png">

Lastly, in order to detect the individual letters, all of the contours of the image are sorted and according to its width and height, determines where the letters are. The following segment of code achieves this step. [2]

``` Python
for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 0.5<=ratio<=5: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
```

The result is

<img src="/Images/grab_digit_contour.png">


### &nbsp;&nbsp;&nbsp; Part 3: Training the Computer with different Networks and using them to detect the letters.
&nbsp;&nbsp;&nbsp; Training the computer takes a long time and memory. First, researched about what types of networks that could be used with the frameword was conducted and MobileNet, ResNet, and Xception were the three models that were compatible. With the training data provided by Nguyen, the program was able to train all of these networks with notable results.

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MobileNet:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This is the network that the framework came with. Even though there was a lot of errors and it took me a long time to get it running, it was able to produce some of the results that the original author had. MobileNet uses lightweight deep convolutional neural networks to provide an efficient model for mobile and embedded vision applications. The following function uses the MobileNet structure to start training from the dataset. [3] [4]

``` Python
# Create our model with pre-trained MobileNetV2 architecture from imagenet
def create_model(lr=1e-4,decay=1e-4/25., training=False,output_shape=y.shape[1]):
    baseModel = MobileNetV2(weights="imagenet", 
                            include_top=False,
                            input_tensor=Input(shape=(80, 80, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(output_shape, activation="softmax")(headModel)
    
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    if training:
        # define trainable lalyer
        for layer in baseModel.layers:
            layer.trainable = True
        # compile model
        optimizer = Adam(lr=lr, decay = decay)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])    
        
    return model
```
The following shows the MobileNet Architecture

<img src="/Images/mobilenet.png">

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ResNet:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ResNet worked flawlessly within the network but it did took a lot longer than MobileNet on Google Colab. ResNet uses residual learning and builds pyramidal cells so that layers could be skipped. [5]

The following shows the ResNet Architecture

<img src="/Images/ResNet.png">


#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Xception:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Xception took around the same time as ResNet. Xception is a deeper implementation of Inception which uses Deapthwise Separable Convolutions to train the data. [6]

The following shows the Xception Architecture

<img src="/Images/Xception.png">

### &nbsp;&nbsp;&nbsp; Challenges:
&nbsp;&nbsp;&nbsp; Throughout this whole project, many problems was encountered. First of all, I discovered that Google Colab was really slow in running the models and tests so I decided to use my own computer with the gpu to hopefully make the training go faster. Then I discovered that even with my gpu, the computer still does not run as fast as Google Colab and my memory was running out. In the end, I switched back to Google Colab and tested the rest of the project there. When using the different networks, problems also occured when chosing which version of the librariesare used such as for the keras application and tensorflow. Throughout the process, there was many bugs as well that was discovered. I was able to overcome all the challenges and tested and experienced with machine learning in deapth. 

### &nbsp;&nbsp;&nbsp; Results:
&nbsp;&nbsp;&nbsp; Here is the final result after the letters are separated and the model has been used to detect the letters.

<img src="/Images/final_result1.png">

## Conclusion:
Working with programs and code can be very frustrating at times but eventually, everything can be solved. Through this project and research, I know many of the weakness and flaw of the original framework. I can do more research and find a better solution to extract the license plates more effectively and also can improve on letter detection as well as the networks.

## References
[1] Detect and Recognize Vehicle’s License Plate with Machine Learning and Python — Part 1: Detection License Plate with Wpod-Net. https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922

[2] Detect and Recognize Vehicle’s License Plate with Machine Learning and Python — Part 2: Plate character segmentation with OpenCV. https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-2-plate-de644de9849f

[3] Detect and Recognize Vehicle’s License Plate with Machine Learning and Python — Part 3: Recognize plate license characters with OpenCV and Deep Learning. https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-3-recognize-be2eca1a9f12

[4] Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017). https://arxiv.org/abs/1704.04861

[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778). https://arxiv.org/abs/1512.03385

[6] Chollet, François. "Xception: Deep learning with depthwise separable convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. https://arxiv.org/abs/1610.02357
