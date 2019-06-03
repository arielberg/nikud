# Hebrew autmatic vocalization marker 
(nikud/ניקוד)

This project is designed to create an AI module to automatic assign vocalization marks to 
unmarked text = (ספר -> סֵפֶר)

The model is using fasttxt as TextToVec trained by wikipedia and 
it is trained mainly on marked text from the bible (more relevant marked text is needed)

The repository main files:
 
* prepareData.py - prepare data from all text in the row folder to a csv file ready for training
* train.py - train the model using the csv file
