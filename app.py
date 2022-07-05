import os
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, flash, redirect, render_template, request, session
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array


# ====== FLASK SETUP ======

UPLOAD_FOLDER = 'C:\\Users\\ASUS\\orbit-belajar-python\\projek-akhir\\KAMI\\static\\upload\\images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'ini secret key KAMI'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ====== Prediction ====== 

model = load_model('model/foodclassification.hdf5')
classes = ['Ayam Panggang', 'Gurami Bakar', 'Nasi Kebuli Kambing', 'Rendang Sapi', 'Tahu Isi', 'Telur Balado', 'Tempe Orek', 'Tumis Udang']

def finds():
  test_datagen = ImageDataGenerator(rescale = 1./255)
  test_dir = 'C:\\Users\\ASUS\\orbit-belajar-python\\projek-akhir\\KAMI\\static\\upload'
  test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size =(224, 224),
        color_mode ="rgb",
        shuffle = False,
        class_mode = None,
        batch_size = 1)

  pred = model.predict_generator(test_generator)
  print(pred)
  
  return str(classes[np.argmax(pred)])

def predict(filename, model):
  img = load_img(filename, target_size = (224, 224))
  img = img_to_array(img)
  img = img.reshape(1, 224, 224, 3)

  img = img.astype('float32')
  img = img/255.0
  result = result = model.predict(img)

  dict_result = {}
  for i in range(8):
    dict_result[result[0][i]] = classes[i]
  
  res = result[0]
  res.sort()
  res = res[::-1]
  prob = res[:3]

  prob_result = []
  class_result = []
  for i in range(3):
    prob_result.append((prob[i]*100).round(2))
    class_result.append(dict_result[prob[i]])
  
  return class_result, prob_result

def food_recipe_nutrition(class_result):
  food_df = pd.read_csv('static/food_db.csv')

  for i in range(len(food_df)):
    if class_result == food_df.loc[i, 'class']:
      new_food_df = food_df.loc[i]
      recipe = new_food_df[1].split(', ')
      steps = new_food_df[2].split(', ')
      nutrition = new_food_df[3].split(', ')

      return recipe, steps, nutrition

# ====== Routes ====== 

@app.route("/")
def home():
  session['secrrt'] = 'sec'
  return render_template("index.html")

@app.route("/about")
def about():
  session['secrrt'] = 'sec'
  return render_template("about.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():
  if request.method == 'GET':
    return render_template("upload.html")
  elif request.method == 'POST':
    if 'inpFile' not in request.files:
      flash('No file part')
      return redirect(request.url)

    file = request.files['inpFile']

    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      img = file.filename
      print(img_path)
      print(img)
      class_result, prob_result = predict(img_path, model)

      predictions = {
        "class1":class_result[0],
        "class2":class_result[1],
        "class3":class_result[2],
        "prob1": prob_result[0],
        "prob2": prob_result[1],
        "prob3": prob_result[2],
      }

      recipe, steps, nutrition = food_recipe_nutrition(class_result[0])
      print("TIPE RECIPE:", type(recipe))

      return render_template("upload.html", result=predictions, recipe=recipe, steps=steps, nutrition = nutrition)
    else:
      error = 'Mohon upload gambar dengan format png, jpg, atau jpeg.'
    
  else:
    return "Unsupported Request Method"


if __name__ == '__main__':
  app.run(port=5000, debug=True)