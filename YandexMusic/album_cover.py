import streamlit as st
from streamlit import caching
from faiss import write_index, read_index
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
index = read_index("YandexMusic/large.index")

#Функция загружет выбранную версию CLIP на имеющийся девайс, на вход получает идентификатор модели согласно каталогу 
def get_model_info(model_ID, device):
    # Save the model to device
	model = CLIPModel.from_pretrained(model_ID).to(device)
 	# Get the processor
	processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)
       # Return model, processor & tokenizer
	return model, processor, tokenizer
	
# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"


# Define the model ID
model_ID = "openai/clip-vit-base-patch32"
# Get model, processor & tokenizer
model, processor, tokenizer = get_model_info(model_ID, device)
#st.cache.clear()
st.title('Определение жанра альбома по картинке')
uploaded_file = st.file_uploader("Загрузите обложку альбома",type=['png','jpg','bmp','tif'])
# Функция принимает на вход картинку, а возвращает эмбеддинг
def get_single_image_embedding(my_image):
    image = processor(
		text = None,
		images = my_image,
		return_tensors="pt"
		)["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    # convert the embeddings to numpy array
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np
# Функция принимает на вход вектор изображения vec, в формета ячейки датафрейма, в которой лежит эмбеддинг, и число  ответов n,а возвращает жанр наиболее часто встречающийся
# среди n обложек альбомов наиболее близких к заданному вектору vec
def find_neighbor(vec,n):
    array_2d_test = vec.reshape(-1, 512)
    D, I = index.search(array_2d_test, n)  # Возвращает результат: Distances, Indices
    #print(I.flatten())
    rez = pd.DataFrame(columns=["genre"])
    for idx in I.flatten():
        rez.loc[ len(rez.index )] = [df.iloc[int(idx)]['genre']]
    return rez.value_counts(normalize=True).head(1).index[0][0]
try:
	image_data = uploaded_file.getvalue()
	img = Image.open(BytesIO(image_data))
	vector = get_single_image_embedding(img)
	df = pd.read_csv('YandexMusic/covers.csv')
	genre = find_neighbor(vector, 7)
	cap = "Вероятный жанр: " +  str(genre)
except: 
	st.write('Ожидаем загрузку обложки')
try:
     st.image(image_data, caption=cap, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
     #st.cache.clear()
except:
     st.write('. . .')

#rec_list=[]
#for i in rage(len(idx)):
#    rec_list.append(df.iloc[i]['cover'])
#st.image(rec_list, caption="Вам могут понравиться") 
