from app import app
from app.models.model import treinar_modelo
import os
from flask import render_template,request, Response, redirect, url_for, jsonify
import cv2
from datetime import datetime
import numpy as np
import time

from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'models', 'modelo_mobilenetv2.h5')

model = None
if os.path.exists(model_path):
    modelo_global = load_model(model_path)
else:
    print("⚠️ Modelo ainda não treinado. A rota de classificação estará indisponível até o treino.")


camera_index = 0
camera = cv2.VideoCapture(camera_index)  # 0 é a câmera padrão do computador

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/capturar', methods=['GET', 'POST'])
def capturar():

# Conta não conformes
    try:
        contadora = [f for f in os.listdir('dataset/Conforme') if os.path.isfile(os.path.join('dataset/Conforme', f))]
        contador_a = len(contadora)
    except FileNotFoundError:
        contador_a = 0

    try:
        contadorb = [f for f in os.listdir('dataset/Naoconforme') if os.path.isfile(os.path.join('dataset/Naoconforme', f))]
        contador_b = len(contadorb)
    except FileNotFoundError:
        contador_b = 0

    return render_template("capturar.html", contador_a=contador_a, contador_b=contador_b)

@app.route('/treinar', methods=['GET', 'POST'])
def treinar():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # pasta app/controllers/.. == app
    dataset_path = os.path.join(base_dir, 'static', 'dataset')
    print("Dataset path:", dataset_path)
    resultado = treinar_modelo(dataset_path)
    modelo_global = load_model(model_path)
    

    history = resultado['history']
    report = resultado['report']
    matriz = resultado['confusion_matrix']

    return render_template(
        'treinar.html',
        acc=history['accuracy'][-1],
        val_acc=history['val_accuracy'][-1],
        loss=history['loss'][-1],
        val_loss=history['val_loss'][-1],
        report=report,
        matriz=matriz
        )

@app.route('/classificar', methods=['GET', 'POST'])
def classificar():
    global modelo_global

    if request.method == 'POST':
        imagem = request.files.get('imagem')
        if not imagem:
            return jsonify({"error": "Imagem não enviada"}), 400

        # Ler imagem para numpy
        file_bytes = np.frombuffer(imagem.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Imagem inválida"}), 400

        # Pré-processamento
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Carregar modelo se necessário
        if modelo_global is None:
            caminho_modelo = os.path.join("app", "models", "modelo_mobilenetv2.h5")
            if not os.path.exists(caminho_modelo):
                return jsonify({"error": "Modelo não treinado ainda"}), 500
            modelo_global = load_model(caminho_modelo)

        # Previsão
        preds = modelo_global.predict(img)
        idx = int(np.argmax(preds))
        classes = ['Conforme', 'Naoconforme']
        resultado = classes[idx]

        return jsonify({"resultado": resultado})

    # GET: mostra a página
    return render_template("classificar.html")

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Codifica o frame em JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Envia o frame como parte do stream contínuo
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capturar/<categoria>', methods=['POST'])
def capturar_categoria(categoria):
    global roi
    success, frame = camera.read()
    if not success:
        return redirect(url_for('capturar'))

    # Se ROI definida, recorta a imagem
    if roi:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]

    # Salva a imagem no dataset
    pasta = f'dataset/{categoria}'
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f'{int(time.time())}.jpg')
    cv2.imwrite(caminho, frame)

    return redirect(url_for('capturar'))

@app.route('/desenhar_area', methods=['POST'])
def desenhar_area():
    imagem = request.files.get('imagem')
    tipo = request.form.get('tipo')

    if not imagem or tipo not in ['conforme', 'nao_conforme']:
        return 'Erro na captura', 400

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pasta_base = os.path.join(base_dir, '..', 'static', 'dataset')

    # Escolhe a subpasta conforme o tipo
    pasta = os.path.join(pasta_base, 'Conforme' if tipo == 'conforme' else 'Naoconforme')

    # Cria a pasta se ela não existir
    os.makedirs(pasta, exist_ok=True)

    # Cria um timestamp para nomear a imagem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Caminho completo para salvar a imagem
    caminho = os.path.join(pasta, f'{tipo}_{timestamp}.jpg')

    # Salva a imagem no caminho determinado
    imagem.save(caminho)

    return 'OK'


@app.route('/trocar_camera', methods=['POST'])
def trocar_camera():
    global camera_index, camera

    camera_index = 1 if camera_index == 0 else 0
    camera.release()
    camera = cv2.VideoCapture(camera_index)

    # Pega o nome da página atual que vem no form
    next_page = request.form.get('next_page', 'capturar')

    return redirect(url_for(next_page))


