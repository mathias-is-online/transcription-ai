# Importa las bibliotecas necesarias
import os
import whisper

# Definir la clase o las funciones que contienen tu lógica
class ANALISIS_AUDIO:
    def __init__(self):
        # Puedes inicializar variables aquí si es necesario
        pass

    def transcribir_audio(self, audio_file_path, language="es"):
        model = whisper.load_model("medium")  # medium  large
        result = model.transcribe(audio_file_path, language=language)
        return result["text"]

    def transcribir_audios_masivo(self, audio_folder, txt_folder):
        print("Hola amigos JEJEJEJE")
        for filename in os.listdir(audio_folder):
            audio_file_path = os.path.join(audio_folder, filename)
            transcription = self.transcribir_audio(audio_file_path)
            print("EL AUDIO FILE PATH QUE SE USA EN LA TRANSCRIPCION ES:  " + audio_file_path)
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_file_path = os.path.join(txt_folder, text_filename)
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                print('Transcripción  : ' + transcription)
                text_file.write(transcription)

# Crear una instancia de la clase
analisis = ANALISIS_AUDIO()

# Definir las rutas a las carpetas de audio y texto
audio_folder = "C:/Users/mathi/Downloads/transcribe/RETENCION_AUDIO"
txt_folder = "C:/Users/mathi/Downloads/transcribe/RETENCION_TEXTO"

# Llamar a la función para transcribir archivos masivos
analisis.transcribir_audios_masivo(audio_folder, txt_folder)
