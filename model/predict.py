import pickle
import pandas as pd

#import model
with open("model/model.pkl","rb") as file:
    model=pickle.load(file)

class_labels=model.classes_.tolist()

def predict_ouput(user_input:dict):
    df=pd.DataFrame([user_input])

    #Predicted class
    predicted_class=model.predict(df)[0]

    #Get probabilities for all the classes
    probabilities=model.predict_proba(df)[0]
    confidence=max(probabilities)

    #Create mapping: {class_name:probability}
    class_probs=dict(zip(class_labels,map(lambda p:round(p,4),probabilities)))

    return {
        "predicted_category":predicted_class,
        "confidence":round(confidence,4),
        "class_probabilities":class_probs

    }