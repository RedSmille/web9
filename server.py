import http.server
import socketserver
import json
import os
import pickle
import numpy as np
import re
import unicodedata
from keras.models import load_model
from respuestas_chatbot import ObtenerRespuesta
import locale

# Configurar locale en español
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    print("La configuración regional 'es_ES.UTF-8' no está disponible.")

# Cargar archivos del chatbot
try:
    with open('Informacion.json', 'r', encoding='utf-8') as archivo:
        Intentos = json.load(archivo)
except Exception as e:
    print(f"❌ Error cargando 'Informacion.json': {e}")
    Intentos = {}

try:
    Palabras = pickle.load(open('words.pkl', 'rb'))
    Clases = pickle.load(open('classes.pkl', 'rb'))
    Modelo = load_model('chatbot_model.keras')
except Exception as e:
    print(f"❌ Error cargando archivos del modelo: {e}")
    exit(1)

# Eliminar acentos y pasar a minúsculas
def NormalizarTexto(Texto):
    Texto = Texto.lower()
    return ''.join(c for c in unicodedata.normalize('NFKD', Texto) if unicodedata.category(c) != 'Mn')

# Tokenización simple sin nltk
def Tokenizar(oracion):
    oracion = NormalizarTexto(oracion)
    return re.findall(r'\b\w+\b', oracion)

# Crear bolsa de palabras
def BolsaDePalabras(oracion):
    palabras_oracion = Tokenizar(oracion)
    bolsa = [1 if palabra in palabras_oracion else 0 for palabra in Palabras]
    return np.array(bolsa)

# Predicción de intención
def PredecirIntencion(oracion):
    bolsa = BolsaDePalabras(oracion)
    resultado = Modelo.predict(np.array([bolsa]))[0]
    UMBRAL_ERROR = 0.25
    resultados = [[i, prob] for i, prob in enumerate(resultado) if prob > UMBRAL_ERROR]
    resultados.sort(key=lambda x: x[1], reverse=True)
    return [{'Intencion': Clases[r[0]], 'Probabilidad': str(r[1])} for r in resultados]

# Configuración del servidor
PUERTO = 8000

class ManejadorChatbot(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        if os.path.exists(self.path[1:]):
            return super().do_GET()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Archivo no encontrado')

    def do_POST(self):
        try:
            longitud = int(self.headers.get('Content-Length', 0))
            datos_post = self.rfile.read(longitud)
            datos = json.loads(datos_post.decode('utf-8'))
            pregunta = datos.get('prompt', '').strip()

            if not pregunta:
                respuesta = {"response": ["Por favor, ingresa un mensaje o pregunta."]}
            else:
                intentos = PredecirIntencion(pregunta)
                texto_respuesta = ObtenerRespuesta(intentos, Intentos)
                respuesta = {"response": texto_respuesta}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(respuesta, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode('utf-8'))

# Iniciar servidor
with socketserver.ThreadingTCPServer(('0.0.0.0', PUERTO), ManejadorChatbot) as httpd:
    print(f'Servidor ejecutándose en: http://localhost:{PUERTO}/')
    httpd.serve_forever()
