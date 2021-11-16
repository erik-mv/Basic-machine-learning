import joblib
import numpy as np

class Classifier(object):
    def __init__(self):
        self.vectorizer = joblib.load("news_vectorizer_dump.pkl")
        self.model = joblib.load("news_model_dump.pkl")
    
    def get_name_by_label(self, label):
        try:
            if abs(label-91.8548838) < 0.001:
                return 'Ваш запрос не похож на множество плохих запросов. \n Либо он хороший, либо не имеет смысла'
            else:
                return '%.2f' % label +'%'
        except:
            return "label error"

    def predict_text(self, text):
        try:
            text = text.split()
            for element in text:
                if element == 'для':
                    text.remove('для')
            text  = ' '.join(text)

            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0] 
        except:
            print("prediction error")
            return None 

    def get_result_message(self, text):
        if len(text.split()) < 3:
            return 'Запрос должен состоять как минимум из 3 слов'
        else:
            prediction = np.expm1(self.predict_text(text))
            return self.get_name_by_label(prediction)
