This read me is to compile notes as a go along my journey of fine tuning my model to best predict values.


# Future Implementations
First and foremost I plan to eventually feed in the data relating to holidays/special observances/ sporting events/ religious events in to my model to help increase accuracy.

## NLP
**[NOTE: This has been added in. please remove/edit this part future me]**I plan on figuring out the best method to convert a set of words to a real number to be able have my model parse things. Hopefully this addition and the bug fixing to ensure it properly works will help in the models sense of accuracy and increase predictions. At the moment there is nothing concrete set for how to go about this however, I am currently considering a bag of words or frequency as a way. They main thing is that i have to considered the uniqueness of the word and if two different groups are connected to one entry. Much to consider for now I will update when I come closer to a solution i like and feel suits this project best.


# Current notes

## model tuning
### This is a giant overview on my findings after running the model throught a few test to summarize how to best run the model(for deeper analysis it can be found in [finding on testing](#findings-on-testing))

As it currently stands the models parameters work best with 3 hidden layers, input layer -> 256 -> a -> b -> 1, where a and b are either the pair(64,32) or (32,4). The model can be split to test, train, and predict in 4 different ways -- Weekend, Weekday, Mornings, and Total data -- to be able to achieve better results on certain areas. 

It is typically set that when working with the weekday and total data to use the parameters of input x 256 x 64 x 32 x 1 or more simply 64x32 at 80-85% training and epochs to be set to 1000. Meanwhile the weekend data is typically better at a 32x4 with 75-85% training data with epochs set to 1000-2000(1000 or 1500 seems to work well). For Morning data it is best to use a 32x4 parameters with 85% training data.

Each one of the three data partitions has a baseline as to how off is can be from the predicted value to be considered correct. At the moment both weekday and total is set to be |y_pred - y_actual| < 10, and weekend is |difference|<15. This is mainly do to the fact that the overall data and weekday tend to have smaller values while weekend has typical values that are 3x that of weekday.

The model runs with an L1 loss function considering its output is positive whole numbers and an adam optimizert with a scheduler to help in decreasing loss over time as the model works.

expanding window - when using expanding windowing using a window of size 5 yields high results as well as 3. >30%.
sliding window - at 20% overlap smaller window does better (45 > 90)points. 90 points doesnt do well, less would be better. small window/large overlap: good for accuracy
                small window/small overlap: fast

_When refering to normal I simply mean that splitting testing and training data down some line and then training and testing the model a handful of times and averaging results._
After implementing sliding and expanding windowing and comparing their testing results I found that the basic average of expanding window, sliding windowing, and normal runs result in better results for weekend days data. I can also achieve similar results by simply taking the average of the weighted ensemble of **windowing only, normal only, and window and normal**. As for the achiving the best results for weekday data it so far appears that using the average of ensemble weighted results of window only, window and sliding, sliding only, sliding and normal, normal only, and normal and expanding can achieve close results to what is to be expected. This can be seen when I had done a few runs by running the model to use certain operational modes only and a mix of the operational modes (the operational modes refered here are running it normal, expanding window, and sliding window). For the time being until further data is gathered and testing is done, I will simply suggest that using the average of the weighted ensemble of predictions to be used as a final prediction for weekday data.
As of this moment nothing has been tested for the morning data everything mentioned in the previous paragraph is intended for the data labeled as data which refers to the afternoons

*The ML.py file can chart the losses and predict values assuming that the necessary values for how far in to predict have been given.



## findings on testing
By opening the parameter_config.xlsx file you can find some of the saved testing that went into finding some of the proper tuning. The train, test, and accuracy tabs show what the different layers where set to to be able to properly optimzize the data set as a whole. with layer 2/(b) going from 32-128 nodes and layer 3/(c) going from 4-64 nodes. 

The tab labeled PCA tab contains 2 sections. The top section contains further testing on the two main model parameters 64x32 and 32x4 on the weekend and weekday data. The bottom half contains the PCA that was ran on each of the entries to find how the train and test losses relate to the models accuracy

the training-data% tab contains the testing to find the best training data amount for weekend data. Considering that the more training data you have the less testing data you have the accuracy can vary hard and thus it should be noted that by increasing the lower training data percentage sets will help increase and find an optimal amount.

# About the files

## Ml.py
This file **WILL** be broken down into more files at a later date to be able to fully document and navigate everything it contains

## extra_data.py
Extra_data is used for gathering and parsing additional outside data that is then tuned for the models inputs or parameters. It contains a way to retrieve weather information, both forecast and historical. This in turn is added the data file for ease of use in the model.The file aslo gathers holiday info that was used to also feed into the model, however the actual implementation of that is still up in the air. The last major thing this file includes is it houses the PCA for the data that was used to determine the minimization of inputs to create a higher accuracy when using certain parameters

## FFT_analysis
FFT_analysis was a file to help use Fourier analysis to help in predicting values in traditional ways. It was also to help me get a different viewpoint to use vizualization to create what would've been hopefully some useful data to compare along with the model. While not perfect it does require some tuning as well to be able to work along side the model in better curating possible future values

## render.py
Render is a simple file that helps vizualize all the data into different ways to see how data is changing. The images are then saved and stored in the image folder to be able to review in futrure cases. This is also where much of the values for output.txt come from given that output is just the text file version of all the images plus some other additional information 

## other files
additional files are supplementary. text goes here