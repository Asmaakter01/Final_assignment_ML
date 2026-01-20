import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("video_game_predict_model.pkl", "rb") as file:
    model = pickle.load(file)
    

def predict_global_sales(Rank, Name, Platform,Year, Genre, Publisher):
    Game_Age = 2026 - Year
    input_df = pd.DataFrame([[
        Rank, Name, Platform,Year, Genre, Publisher, Game_Age
    ]],
    columns= [
        'Rank', 'Name', 'Platform', 'Year','Genre', 'Publisher', 'Game_Age'
    ]                        
    )
    
    prediction = model.predict(input_df)[0]
    return f"Predicted Sales Result: {prediction:.2f} millions"

inputs = [
    gr.Number(label='Rank'),
    gr.Text(label='Name'),
    gr.Text(label='Platform'),
    gr.Number(label='Year'),
    gr.Text(label='Genre'),
    gr.Text(label='Publisher')
]    


app = gr.Interface(
    fn = predict_global_sales,
    inputs = inputs,
    outputs = 'text',
    title = "Video Game Sales Predictor"
)

app.launch(share=True)