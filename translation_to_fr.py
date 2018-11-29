import numpy as np
import sys
from googletrans import Translator
import pandas as pd

def translation(batch):
    """customer your personal file path """
    file = "/Users/leli/iAdvize/data/training.1600000.processed.noemoticon_translated.csv"
    df = pd.read_csv(file, sep = ",", usecols = ['target', 'id', 'date', 'flag', 'user', 'text', 'translated'], encoding = 'latin-1')
    
    bg = df.iloc[:, -1].last_valid_index() + 1
    
    
    print('begin from {}'.format(bg))
    
    end = bg + int(batch)
    
    for i in range(bg, end):
        
        translator = Translator()
        original = df.iloc[i, -2]
        
        try:
        
            r = translator.translate(original, dest = 'fr')
            df.iloc[i, -1] = r.text
            
        except:
            pass
        
    print("{}".format(df.iloc[(end-1), -1]))
    
    df.to_csv(file, sep = ",",
           index = False)
            
    
    
batch = sys.argv[1]

translation(batch) ## translatate batch size english tweets to french.



















