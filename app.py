from flask import Flask, render_template, send_from_directory
import os
import kagglehub

app = Flask(__name__)

@app.route("/download-dataset")
def download_dataset():
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    return f"Dataset descargado en: {path}"

@app.route("/")
def home():
    try:
        dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        return render_template('index.html', dataset_path=dataset_path, folders=folders)
    except Exception as e:
        return f"Error al cargar el dataset: {str(e)}. Usa /download-dataset para descargarlo primero."

@app.route("/folder/<path:folder_path>")
def show_folder(folder_path):
    try:
        dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        full_path = os.path.join(dataset_path, folder_path)
        
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return "Carpeta no encontrada"
        
        items = os.listdir(full_path)
        files = [f for f in items if os.path.isfile(os.path.join(full_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        subfolders = [f for f in items if os.path.isdir(os.path.join(full_path, f))]
        
        return render_template('index.html', 
                              dataset_path=dataset_path, 
                              current_folder=folder_path,
                              files=files,
                              folders=subfolders)
    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/image/<path:image_path>")
def serve_image(image_path):
    try:
        dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        directory = os.path.dirname(os.path.join(dataset_path, image_path))
        filename = os.path.basename(image_path)
        return send_from_directory(directory, filename)
    except Exception as e:
        return f"Error al cargar la imagen: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)