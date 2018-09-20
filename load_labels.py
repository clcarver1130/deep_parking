def load_labels(): 
    df = pd.read_csv('training_labels.csv') 
    labels = df[Label] 
    return labels