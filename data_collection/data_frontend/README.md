app.py:

Easily manage your training data with this Streamlit app to browse multiple RC car run folders, view frames, visualize controls, and delete bad frames (removes image file + CSV row). Also has a Model Prediction page where you can upload a model weights file, an image to predict, and specify the resnet architecture and receive the model's prediction for that image. Install dependencies using requirements.txt.

Run: ```streamlit run app.py```


![demo](https://cdn.discordapp.com/attachments/1146243338832457751/1447347533746802829/Screenshot_29.png?ex=69374aee&is=6935f96e&hm=5206c0e4390c59ac9f297cfbe0399db17b9db31921cd322a88f4cf007c102b91)

dataset_csv_creator.py: Takes a folder filled with runs recorded while driving the RC Car and combines the image with the csv data into one csv. The csv format is timestamp,steer_us,throttle_us,steer_norm,throttle_norm,depth_front,R1,G1,B1...

augment_data.py: Performs several augmentations on the data individually, as well as a "combination augmentation" step which combines several augmentations with 25-75% intensity. Specify the dataset csv.

augment_data_gpu.py: Same augmentations as augment_data.py but different approach, applies augmentations in batches on GPU using Pytorch and Kornia to leverage the GPU to make augmentations faster. Install dependencies using requirements_gpu.txt.

Dockerfile / docker-compose.yml available for containerizing frontend for easy deployment.
