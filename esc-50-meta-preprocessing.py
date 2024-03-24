import pandas as pd

if __name__ == "__main__":
    csv = pd.read_csv("ESC-50-master/meta/esc50.csv")
    df = pd.DataFrame(columns=['label', 'path'])
    df['label']=csv['target']
    df['category']=csv['category']
    df['path']="ESC-50-master/audio/"+csv['filename']
    df.to_csv("esc-50-data.csv")