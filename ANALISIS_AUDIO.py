import warnings
from datetime import datetime
import os
import whisper
import cx_Oracle
from transformers import MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer  # 64 bits
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import csv
import chardet
import Levenshtein as lev
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import spacy
from collections import Counter
from spacy.lang.es.stop_words import STOP_WORDS
 
class ANALISIS_AUDIO:
      
    def __init__(self, user, password):
        warnings.filterwarnings('ignore')
        self.llamada_call_min_sent_oraciones = 5
        self.llamada_call_min_sent_frases = 10
        self.llamada_call_min_sent_palabras = 20
        self.llamada_call_min_toxicidad_oraciones = 5
        self.llamada_call_min_toxicidad_frases = 10
        self.llamada_call_min_toxicidad_palabras = 20

        self.wsp_min_sent_oraciones = 5
        self.wsp_min_sent_frases = 10
        self.wsp_min_sent_palabras = 20
        self.wsp_min_toxicidad_oraciones = 5
        self.wsp_min_toxicidad_frases = 10
        self.wsp_min_toxicidad_palabras = 20

        self.user= user
        self.password = password  
        self.lisuras =[
    "carajo", "mierda", "joder", "puta", "cojudo", "huevón", "chingar",
    "pendejo", "cabrón", "gil", "conchatumadre", "marica", "poto", "cagar",
    "pinga", "maldito", "idiota", "imbécil", "estúpido", "tarado", "malparido",
    "conchesumadre", "putamadre", "reputamadre", "hijueputa", "mamahuevo",
    "cagada", "chingada", "jodida", "putada", "webadas", "spam", "molesta", "conchetumare"
    ]
        self.frases_portout= ["me voy a cambiar","movistar", "entel", "bitel","otro operador", "de otro lado me estan llamando",
                              "de otro me estan llamando",
                       "otra empresa", "portar", "voy a portar", "desvincular", "nunca más", "cambiar a"]
        
        self.frases_reno= ["equipo","celular","otro equipo", "nuevo celular", "nuevo equipo", "renovar", "quiero renovar", "deseo otro equipo", "cambiar equipo"]

        self.frases_no_contact= ["ya no envien mensajes", "no quiero que me envien", "no quiero nada", "dejen de molestar", "dejen de mandar", "basta de ",
                                  "paren de mandar", "ya no esten", "por favor ya no", "estoy arto", "hostigamiento", "favor no enviar", "no enviar"
                                  "dejen de enviar", "no soy", "spam", "nada de claro", "yo no soy el titular", "no soy el titular"]
        
        self.frases_insatisfaccion= ["no me gusta", "no cumple", "no funciona", "mala experiencia",  "muy malo", "decepcion", "lento", " falla ", "problema",
                                         "carajo", "mierda", " joder ", " puta ", "cojudo", "huevón", "chingar",
    "pendejo", "cabrón", " gil ", "conchatumadre", " marica ", " poto ", " cagar ",
    " pinga ", "maldito", "idiota", "imbécil", "estúpido", "tarado", "malparido",
    "conchesumadre", "putamadre", "reputamadre", "hijueputa", "mamahuevo",
    "cagada", "chingada", "jodida", "putada", "webadas", " spam ", "molesta", "conchetumare"]
 
    def transcribir_audio(self,audio_file_path, language="es"):
        model = whisper.load_model("medium")#medium  large
        result = model.transcribe(audio_file_path, language=language)
        return result["text"]
 
    def transcribir_audios_masivo(self,audio_folder, txt_folder):
        print("ola amigos JEJEJEJE")
        for filename in os.listdir(audio_folder):
            #if filename.endswith(".mp3") or filename.endswith(".wav"):
            audio_file_path = os.path.join(audio_folder, filename)
            transcription = self.transcribir_audio(audio_file_path)
            print("EL AUDIO FILE PATH QUE SE USA EN LA TRANSCRIPCION ES:  " + audio_file_path)
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_file_path = os.path.join(txt_folder, text_filename)
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                print('transcripcion  : ' + transcription)
                text_file.write(transcription)

    
   
    def traducir_a_ingles(self,texto):
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")
        encoded = tokenizer.encode(texto, return_tensors="pt", max_length=512, truncation=True)
        translated = model.generate(encoded)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
 
    def resumir_con_t5(self,texto):
        modelo = "t5-large"  # O "t5-base"
        tokenizer = T5Tokenizer.from_pretrained(modelo)
        model = T5ForConditionalGeneration.from_pretrained(modelo)
        inputs = tokenizer.encode("summarize: " + texto, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
    def traducir_a_espanol(self,texto):
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        encoded = tokenizer.encode(texto, return_tensors="pt", max_length=512, truncation=True)
        translated = model.generate(encoded)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def resumir_texto(self, texto_original, tipo=0):
        resumen_final=""
        if tipo==0:
            texto_en_ingles = self.traducir_a_ingles(texto_original)
            resumen_en_ingles = self.resumir_con_t5(texto_en_ingles)
            resumen_final = self.traducir_a_espanol(resumen_en_ingles)
        else :
            texto_en_ingles = self.traducir_a_ingles(texto_original)
            resumen_en_ingles = self.resumir_con_bart(texto_en_ingles)
            resumen_final = self.traducir_a_espanol(resumen_en_ingles)
        return resumen_final
    
    
    def resumir_con_bart(texto):
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
 
        inputs = tokenizer([texto], max_length=1024, return_tensors='pt')
 
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=10,  # Aumentar el número de beams
            max_length=100,  # Extender la longitud máxima
            min_length=100,  # Establecer una longitud mínima significativa
            length_penalty=2.5,  # Incrementar la penalización por longitud
            no_repeat_ngram_size=4,  # Evitar la repetición de n-gramas
            early_stopping=False  # Desactivar el early stopping
            )
 
        resumen_en_ingles = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
        return resumen_en_ingles
 
    
    def analyze_sentiment_es(self,texto):
   
        model_name = "finiteautomata/beto-sentiment-analysis"  
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        results = sentiment_pipeline(texto)

        return results
    

     
    def cargar_modelo_analisis_sentimientos_distilbert(self):
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        return sentiment_pipeline
 
    def analizar_sentimiento_distilbert(self,texto, sentiment_pipeline):
        resultados = sentiment_pipeline(texto)
        return resultados
 
    def mostrar_analisis_sent(self, texto):
        sentiment_pipeline_distilbert = self.cargar_modelo_analisis_sentimientos_distilbert()
        resultados = self.analizar_sentimiento_distilbert(self.traducir_a_ingles(texto), sentiment_pipeline_distilbert)
        return resultados
 
    def detectar_toxicidad(self,texto):
        modelo = "unitary/toxic-bert"
        tokenizer = AutoTokenizer.from_pretrained(modelo)
        model = AutoModelForSequenceClassification.from_pretrained(modelo)
        toxicity_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
        resultados = toxicity_pipeline(self.traducir_a_ingles(texto))
        return resultados


    def eliminar_repetidos_excesivos(self, texto, limite=5):
        palabras = texto.split()
        palabras_procesadas = []
 
        contador_repetidos = 1
        palabra_anterior = None
 
        for palabra in palabras:
            if palabra == palabra_anterior:
                contador_repetidos += 1
                if contador_repetidos > limite:
                    continue
            else:
                contador_repetidos = 1
            palabras_procesadas.append(palabra)
            palabra_anterior = palabra
 
        return ' '.join(palabras_procesadas)




    def llamada_call_procesar_transcripcion(self, txt_folder):
        longitud_maxima_caracteres = 1000
        #nltk.download('punkt')
        spanish_tokenizer = PunktSentenceTokenizer()
        for filename in os.listdir(txt_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(txt_folder, filename)
 
                with open(file_path, 'rb') as rawfile:
                    result = chardet.detect(rawfile.read())
                    encoding = result['encoding']
 
                with open(file_path, 'r', encoding=encoding) as file:
                    content =self.eliminar_repetidos_excesivos(file.read())
                    
 
                oraciones = spanish_tokenizer.tokenize(content)
 
                csv_filename = filename.replace('.txt', '.txt')
                csv_file_path = os.path.join(txt_folder, csv_filename)
               
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for oracion in oraciones:
                        palabras = oracion.split(" ")
                        indice_inicio = 0  
                        while indice_inicio < len(palabras):
                            oracion_nueva = ""
                            longitud_actual = 0

                            for i in range(indice_inicio, len(palabras)):
                                longitud_potencial = longitud_actual + len(palabras[i]) + (1 if longitud_actual > 0 else 0)
 
                                if  longitud_potencial > longitud_maxima_caracteres:
                                    print("longitud oracion maxima superada  " + csv_filename)

                                    break
 
                                oracion_nueva += (" " if longitud_actual > 0 else "") + palabras[i]
                                longitud_actual = longitud_potencial
                                indice_inicio = i + 1
 
                            csv_writer.writerow([oracion_nueva])




                            """ with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    #csv_writer.writerow(["ORACIONES"])  
                    for oracion in oraciones:

                        #agregar for junto con split " " para contar palabra por palabra, 
                        # mientras no supere 2k ir concatenando, 
                        # si lo supera insertar oracion acumulada y concatenar otra de la misma manera
                        palabras = oracion.split(" ")
                        cantidad_grupos = round((len(oracion)/longitud_maxima_caracteres) + 0.5)
                        oracion_nueva= ""
                        var_cantidad=0
                        contador_palabras = 1

                        if cantidad_grupos==1:
                            csv_writer.writerow([oracion])
                            
                        else :
                            print("longitud oracion maxima superada  " + str(cantidad_grupos) + "grupos  :  " + csv_filename)
                            for i in range(cantidad_grupos):
                                oracion_nueva=""
                                for orden, palabra in enumerate(palabras, start = contador_palabras):
                                    if len(oracion_nueva)>=longitud_maxima_caracteres : break
                                    oracion_nueva = oracion_nueva + " " + palabra
                                    contador_palabras = contador_palabras + 1

                                csv_writer.writerow([oracion_nueva])"""

             


    def find_similar_phrases(self,text, phrases, max_distance=2, modo=0):
        max_distance=1
        words = text.split()
        encontradas = []
        frases_buscar=[]
        saludo  = ["Buenos días"]
        ofrecimiento = ["super promoción", "promoción", "beneficios", "ofrecerte", "tenemos una excelente promoción", "sido seleccionada", 
                        "otorgarle grandes beneficios", "ofrecerte mejores beneficios", 
                        "Estás calificando para un", "ha sido calificada para poder acceder", 
                        "llamándote para comentar acerca", "bono", "bonificación", "por migrar", "ofreciendo"]
                
        venta =["valídame tu nombre completo y tus apellidos","solo validaremos tus datos", 
                "nombre y apellidos completos",
                "¿estaria de acuerdo?",
                "¿estás de acuerdo?",
                "validarme su",
                "validarme sus",
                "validaremos que",
                "¿nos brinda", "en su próxima entrega de saldo ¿estás de acuerdo?", 
                "sólo necesitamos validar tus datos", 
                "necesitariamos el nombre y apellidos completos",
                "¿Cuáles son sus nombres y apellidos completos?", 
                "¿Cuál es su número de DNI?", "¿Cuál es su fecha de nacimiento?",
                "¿Nos validas tus datos para realizar?", "validar sus nombres y apellido"]
        despedida = ["estamos procediendo a realizar" , "eso seria todo",  "medianoche estará recibiendo",
                     "medianoche usted estará recibiendo",
                     "dentro de las 24 horas", "dentro de 24 horas",
                     "bienvenido al nuevo plan","bienvenido a tu nuevo plan",
                     "contará con el servicio el día de mañana", "servicio se activará", "contará con el servicio", 
                     "servicio se activará en un plazo de", "estamos procediendo a realizar la migración de su plan al", "estaría activando" ]

	
        if modo==0:
            frases_buscar = phrases
        elif modo ==1:
            frases_buscar = ofrecimiento
        elif modo ==2:
            frases_buscar = venta
        elif modo ==3:
            frases_buscar = despedida
        elif modo ==4:
            frases_buscar = self.frases_portout
        elif modo ==5:
            frases_buscar = self.frases_reno
        elif modo ==6:
            frases_buscar = self.frases_no_contact
        elif modo ==7:
            frases_buscar = self.frases_insatisfaccion



        for phrase in frases_buscar:
            phrase_length = len(phrase.split())
            for i in range(len(words) - phrase_length + 1):
                segment = " ".join(words[i:i + phrase_length])
                distance = lev.distance(segment, phrase)
                if distance <= max_distance:
                    encontradas.append((phrase, segment, distance, i))
        return encontradas
 


 
    def llamada_call_procesar_conversacion_oracle(self, txt_folder, phrases_to_find, campaña, n_frases=4):
        conn = self.connectionDWO()
        cursor = conn.cursor()
        global_text=""
        contenido= ""
        
        for filename in os.listdir(txt_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(txt_folder, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    contenido = file.read()

                with open(file_path, 'r', encoding='utf-8') as file:
                    conversacion = ""
                    num_etapa=0
                    etapa_ofrecimiento = False
                    etapa_venta = False
                    etapa_despedida = False
                    id_llamada = str(filename).rstrip(".txt")
                    for orden, frase in enumerate(file, start=1):
                        #conversacion = conversacion + " " + frase
                        frase_encontrada=""  
                        frase_ofrecimiento = ""
                        frase_venta = ""
                        frase_despedida = ""
                        toxicidad=""
                        sentimiento = ""
                        pregunta_encontrada = ""

                        frase_portout =""  
                        frase_reno=""
                        frase_no_contact=""
                        frase_insatisfaccion=""
                        
                        
                        frase = frase.strip()
                        found_frase_encontrada = self.find_similar_phrases(frase, phrases_to_find,1,0)
                        if found_frase_encontrada:
                            frase_encontrada = str(found_frase_encontrada)
                            #toxicidad= str(self.detectar_toxicidad(frase)[0]['score'])
                            #sentimiento = str(self.mostrar_analisis_sent(frase))


    
                        found_frase_ofrecimiento = self.find_similar_phrases(frase, phrases_to_find,2,1)
                        found_frase_venta = self.find_similar_phrases(frase, phrases_to_find,2,2)
                        found_frase_despedida = self.find_similar_phrases(frase, phrases_to_find,2,3)

                  

                        found_frase_portout = self.find_similar_phrases(frase, phrases_to_find,2,4)
                        found_frase_reno = self.find_similar_phrases(frase, phrases_to_find,2,5)
                        found_frase_no_contact = self.find_similar_phrases(frase, phrases_to_find,2,6)
                        found_frase_insatisfaccion = self.find_similar_phrases(frase, phrases_to_find,2,7)


                        if found_frase_portout:frase_portout = str(found_frase_portout[0])  #portout
                        if found_frase_reno:frase_reno = str(found_frase_reno[0])  #reno
                        if found_frase_no_contact:frase_no_contact = str(found_frase_no_contact[0])  # no contact
                        if found_frase_insatisfaccion:frase_insatisfaccion = str(found_frase_insatisfaccion[0])  # no contact



                        if self.detectar_pregunta_frase(frase): pregunta_encontrada = "pregunta"
                    
                        if found_frase_ofrecimiento:
                            if not etapa_ofrecimiento and num_etapa == 0:
                                etapa="OFRECIMIENTO" 
                                etapa_ofrecimiento = True
                                num_etapa= num_etapa +1

                            #frase_ofrecimiento = str(found_frase_ofrecimiento[0]) 
                            frase_ofrecimiento = str(found_frase_ofrecimiento)          

                        if found_frase_venta:
                            if not etapa_venta and num_etapa == 1 :
                                etapa="VENTA" 
                                etapa_venta = True
                                num_etapa= num_etapa +1
                            #frase_venta = str(found_frase_venta[0])
                            frase_venta = str(found_frase_venta)


                        if found_frase_despedida:
                            if not etapa_despedida and num_etapa == 2 :
                                etapa="DESPEDIDA" 
                                etapa_despedida = True
                                num_etapa= num_etapa +1
                            #frase_despedida = str(found_frase_despedida[0]) 
                            frase_despedida = str(found_frase_despedida) 


                        if num_etapa ==1 or num_etapa ==2  :
                            pass
                            #toxicidad=self.detectar_toxicidad(frase)[0]['score']
                            #sentimiento = self.mostrar_analisis_sent(frase)
                        if num_etapa ==3  :
                            pass
                            #sentimiento = self.mostrar_analisis_sent(frase)

 
                        #toxicidad=str(self.detectar_toxicidad(frase)[0]['score'])
                        #sentimiento = str(self.mostrar_analisis_sent(frase))
                        cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_DET (ID, ORDEN, MENSAJE, FRASE_ENCONTRADA,FRASE_OFRECIMIENTO,FRASE_VENTA,FRASE_CIERRE_VENTA,PREGUNTA_ENCONTRADA, frase_portout, frase_reno, frase_no_contact, frase_insatisfaccion, TOXICIDAD, SENTIMIENTO, CAMPAÑA ) VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15)", 
                                   (id_llamada, orden, frase, frase_encontrada, frase_ofrecimiento, frase_venta, frase_despedida,pregunta_encontrada,frase_portout,frase_reno,frase_no_contact,frase_insatisfaccion, toxicidad,sentimiento, campaña ))  

                        conn.commit()
                    #print(self.resumir_texto(conversacion))    
                    palabras_ordenadas, lisuras_encontradas, preguntas_ordenadas, frases_encontradas, oraciones_encontradas = self.analisis_conversacion(contenido, self.lisuras, n_frases)  #conversacion
                    #print("Palabras más usadas:", palabras_ordenadas)
                    #print("Lisuras encontradas:", lisuras_encontradas)
                    #print("Preguntas más frecuentes:", preguntas_ordenadas)
                    print("NIVEL CHAT : Frases más frecuentes:", frases_encontradas)


                    try:
                        for orden, (palabra, frecuencia) in enumerate(palabras_ordenadas, start=1):
                            if frecuencia>0:
                                cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS (ID, ORDEN, TIPO, VALOR,FRECUENCIA, CAMPAÑA) VALUES (:1, :2, :3, :4, :5,:6)",
                            (id_llamada, orden, 'PALABRA_FRECUENTE', f'{palabra}',frecuencia,campaña))
 
                        for orden, (lisura, frecuencia) in enumerate(lisuras_encontradas.items(), start=1):
                            if frecuencia>0:
                                cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS (ID, ORDEN, TIPO, VALOR,FRECUENCIA, CAMPAÑA) VALUES (:1, :2, :3, :4, :5,:6)",
                           (id_llamada, orden, 'LISURA', f'{lisura}',frecuencia,campaña))
 
                        for orden, (pregunta, frecuencia) in enumerate(preguntas_ordenadas, start=1):
                            if frecuencia>0:
                                cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS (ID, ORDEN, TIPO, VALOR,FRECUENCIA, CAMPAÑA) VALUES (:1, :2, :3, :4, :5,:6)",
                           (id_llamada, orden, 'PREGUNTA_FRECUENTE', f'{pregunta}',frecuencia,campaña))
                            
                        for orden, (frase, frecuencia) in enumerate(frases_encontradas, start=1):
                            if frecuencia>0:
                                oracion  = ' '.join(frase)
                                cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS (ID, ORDEN, TIPO, VALOR,FRECUENCIA, CAMPAÑA) VALUES (:1, :2, :3, :4, :5,:6)",
                                (id_llamada, orden, 'FRASE_FRECUENTE', f'{oracion}',frecuencia,campaña))
                  
                        for orden, (oracion_o, frecuencia) in enumerate(oraciones_encontradas, start=1):
                            if frecuencia>1:
                                oracion  = ''.join(oracion_o)
                                cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS (ID, ORDEN, TIPO, VALOR,FRECUENCIA, CAMPAÑA) VALUES (:1, :2, :3, :4, :5,:6)",
                                (id_llamada, orden, 'ORACION_FRECUENTE', f'{oracion}',frecuencia,campaña))
                  
 
                        conn.commit()
                    except cx_Oracle.DatabaseError as e:
                        print(f'Error al insertar en la base de datos: {e}')
                    finally:
                        pass

                global_text= global_text + ".\n " + contenido # conversacion

        #print("CONVERSACION GLOBAL")
        #print(global_text)
        palabras_ordenadas, lisuras_encontradas, preguntas_ordenadas, frases_encontradas, oraciones_encontradas = self.analisis_conversacion(global_text, self.lisuras,n_frases)
       
        print("GLOBAL :")
        #print("Palabras más usadas:", palabras_ordenadas)
        #print("Lisuras encontradas:", lisuras_encontradas)
        #print("Preguntas más frecuentes:", preguntas_ordenadas)
        print("Frases más frecuentes:", frases_encontradas)
        print("Oraciones más frecuentes:", oraciones_encontradas)

        try:
            toxicidad=""
            sentimiento = ""
            for orden, (palabra, frecuencia) in enumerate(palabras_ordenadas, start=1):
                if frecuencia>5:

                    cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS_GLOBAL (TIPO, VALOR, CANTIDAD, CAMPAÑA, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6)",
                            ('PALABRA_FRECUENTE', f'{palabra}',frecuencia,campaña,sentimiento, toxicidad))
                    
            for orden, (lisura, frecuencia) in enumerate(lisuras_encontradas.items(), start=1):
                
                cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS_GLOBAL (TIPO, VALOR, CANTIDAD,CAMPAÑA, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6)",
                            ('LISURA', f'{lisura}',frecuencia,campaña,sentimiento, toxicidad))
                    
            for orden, (pregunta, frecuencia) in enumerate(preguntas_ordenadas, start=1):
                
                cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS_GLOBAL (TIPO, VALOR, CANTIDAD,CAMPAÑA, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6)",
                            ('PREGUNTA_FRECUENTE', f'{pregunta}',frecuencia,campaña,sentimiento, toxicidad))  
                                      
            for orden, (frase, frecuencia) in enumerate(frases_encontradas, start=1):
                toxicidad=""
                sentimiento = ""
                if frecuencia>1:
                    oracion  = ' '.join(frase)
                    cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS_GLOBAL (TIPO, VALOR, CANTIDAD,CAMPAÑA, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6)",
                            ('FRASE_FRECUENTE', f'{oracion}',frecuencia,campaña, sentimiento, toxicidad))
        
            for orden, (oracion_c, frecuencia) in enumerate(oraciones_encontradas, start=1):
                toxicidad=""
                sentimiento = ""
                oracion  = ''.join(oracion_c)
                if frecuencia>5:
                    #toxicidad=str(self.detectar_toxicidad(oracion)[0]['score'])
                    #sentimiento = str(self.mostrar_analisis_sent(oracion))
                    pass

                try :
                    if frecuencia>1:cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS_GLOBAL (TIPO, VALOR, CANTIDAD,CAMPAÑA, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6)",
                            ('ORACION_FRECUENTE', f'{oracion}',frecuencia,campaña, sentimiento, toxicidad))
        

                except cx_Oracle.DatabaseError as e:
                    cursor.execute("INSERT INTO C26796_CONVERSACION_CALL_ANALISIS_GLOBAL (TIPO, VALOR, CANTIDAD,CAMPAÑA, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6)",
                            ('ORACION_FRECUENTE', f'{oracion[:3900]}',frecuencia,campaña, sentimiento, toxicidad))
        
                    print(f'Error oracion muy larga') 
                
            
            conn.commit()
        except cx_Oracle.DatabaseError as e:
                        print(f'Error al insertar en la base de datos: {e}')        
        
        conn.commit()
        cursor.close()
        conn.close()


    def limpiar_palabras(self,palabras):
        palabras_limpia = [palabra.lower() for palabra in palabras if palabra.isalpha()]
        return palabras_limpia
 
    def obtener_oraciones(self,texto):
        oraciones = sent_tokenize(texto, language='spanish')
        return oraciones

    def encontrar_frases_comunes_N(self,texto, n=6):

        nlp = spacy.load("es_core_news_sm")
        doc = nlp(texto)
        ngrams = self.obtener_ngrams(doc, 3)

        frecuencia_frases = Counter(ngrams)
        print("TEXTO  : " + texto)
        print(frecuencia_frases.most_common())
        return frecuencia_frases.most_common()

    """frases_comunes = encontrar_frases_comunes(texto, n=3)  
        for frase, frecuencia in frases_comunes:
            print(f"Frase: '{frase}', Frecuencia: {frecuencia}")"""



    def encontrar_frases_comunes_F(self,texto, n=4):
        nlp = spacy.load("es_core_news_sm")
        texto = nlp(texto)
        oraciones = self.obtener_oraciones(texto)
        todas_las_frases = []


 
        for oracion in oraciones:
            palabras = word_tokenize(oracion, language='spanish')
            palabras = self.limpiar_palabras(palabras)
            #frases = ngrams(palabras, n)
            frases = self.obtener_ngrams(palabras, n)
            todas_las_frases.extend(frases)

        
        
        

        frecuencia_frases = Counter(ngrams)
        print("TEXTO  : " + texto)
        print(frecuencia_frases.most_common())
        return frecuencia_frases.most_common()

    """frases_comunes = encontrar_frases_comunes(texto, n=3)  
        for frase, frecuencia in frases_comunes:
            print(f"Frase: '{frase}', Frecuencia: {frecuencia}")"""

    #metodo antes : encontrar_frases_comunes
    def encontrar_frases_comunes(self,texto, n=2):
        oraciones = self.obtener_oraciones(texto)
        todas_las_frases = []


 
        for oracion in oraciones:
            palabras = word_tokenize(oracion, language='spanish')
            palabras = self.limpiar_palabras(palabras)
            frases = ngrams(palabras, n)
            todas_las_frases.extend(frases)
          
        frecuencia_frases = Counter(todas_las_frases)
        return frecuencia_frases.most_common()
    


    
    
    def limpiar_token(self,token):
        return token.text.lower().strip()
    def es_token_valido(self,token):
        return token.text not in STOP_WORDS and token.text not in string.punctuation
    def obtener_ngrams(self,doc, n=2):
        ngrams = []
        for index in range(len(doc) - n + 1):
            ngram = doc[index:index + n]
            if all(self.es_token_valido(token) for token in ngram):
                ngrams.append(' '.join(self.limpiar_token(token) for token in ngram))
        return ngrams
    


    
    def encontrar_frases_comunes_2(self,texto):
        #oraciones = self.obtener_oraciones(texto)
        todas_las_frases = []


        #NUEVO
        oraciones = nltk.sent_tokenize(texto.lower())
 
        for oracion in oraciones:
            #palabras = word_tokenize(oracion, language='spanish')
            #palabras = self.limpiar_palabras(palabras)
            #frases = ngrams(palabras, n)
            #todas_las_frases.extend(frases)
            pass
 
        todas_las_frases.extend(oraciones)
  
        frecuencia_frases = Counter(todas_las_frases)
        return frecuencia_frases.most_common()
    


    def analisis_conversacion(self,texto, lisuras, largo=4):
        #nltk.download('punkt')
        #nltk.download('stopwords')
        palabras_no_deseadas = set(stopwords.words('spanish') + list(string.punctuation)+ ['``', "''", '..', '...' , 'boton_no', 'boton_si' ])
 

        palabras = nltk.word_tokenize(texto.lower())
        oraciones = nltk.sent_tokenize(texto.lower())
        
        palabras_filtradas = [palabra for palabra in palabras if palabra not in palabras_no_deseadas]
 
        frecuencia_palabras = nltk.FreqDist(palabras_filtradas)
        lisuras_encontradas = {palabra: freq for palabra, freq in frecuencia_palabras.items() if palabra in lisuras}
 
        preguntas = [oracion for oracion in oraciones if oracion.endswith('?')]
        frecuencia_preguntas = Counter(preguntas)
 
        palabras_ordenadas = sorted(frecuencia_palabras.items(), key=lambda x: x[1], reverse=True)
        preguntas_ordenadas = sorted(frecuencia_preguntas.items(), key=lambda x: x[1], reverse=True)

        frases_comunes = self.encontrar_frases_comunes(texto, n=largo)
        oraciones_comunes = self.encontrar_frases_comunes_2(texto)          
 
        return palabras_ordenadas, lisuras_encontradas, preguntas_ordenadas, frases_comunes,oraciones_comunes



    
    def detectar_pregunta_frase(self,texto):
        #nltk.download('punkt')
        #nltk.download('stopwords')
        oraciones = nltk.sent_tokenize(texto.lower())
        
     
        preguntas = [oracion for oracion in oraciones if oracion.endswith('?')]
        
        return preguntas
 
     




    def WSP_UPG_analizar_conversacion(self, frases_obj ):

        query_TOTAL="""SELECT A.NUMEROTICKET, B.CONTENIDO_INTERACCION, B.ENTE_COMUNICADOR, B.FECHA_INTERACCION, 
B.HORA_INTERACCION, B.ROWNUM_ID, A.FLAG_ETAPA, A.FLAG_ACCION, 
CAST(NULL AS VARCHAR2(100) ) AS  FRASE_ENCONTRADA ,
CAST(NULL AS NUMBER) AS TOXICIDAD, CAST(NULL AS VARCHAR2(100) ) AS  SENTIMIENTO 
FROM C26796_TEMP_UPGRADE_WSP_ETAPAS A
LEFT JOIN temp_wsp_det B ON A.NUMEROTICKET= B.NUMEROTICKET AND
TRUNC(B.FECHA_INTERACCION) BETWEEN TO_DATE(A.FECHA_INICIO, 'DD/MM/YYYY') 
AND TO_DATE(A.FECHA_FIN, 'DD/MM/YYYY')
WHERE A.FLAG_ETAPA='MENSAJE INI. CAMPAÑA' 
AND  B.ENTE_COMUNICADOR = 'PERSONA'
AND B.FECHA_INTERACCION >= TO_DATE('01/06/2023', 'DD/MM/YYYY')
ORDER BY A.NUMEROTICKET, B.FECHA_INTERACCION, B.HORA_INTERACCION, B.ROWNUM_ID"""

        query1="""SELECT A.NUMEROTICKET, B.CONTENIDO_INTERACCION, B.ENTE_COMUNICADOR, B.FECHA_INTERACCION, 
B.HORA_INTERACCION, B.ROWNUM_ID, A.FLAG_ETAPA, A.FLAG_ACCION, CAST(NULL AS VARCHAR2(100) ) AS  FRASE_ENCONTRADA ,
CAST(NULL AS NUMBER) AS TOXICIDAD, CAST(NULL AS VARCHAR2(100) ) AS  SENTIMIENTO 
FROM C26796_TEMP_UPGRADE_WSP_ETAPAS A
LEFT JOIN temp_wsp_det B ON A.NUMEROTICKET= B.NUMEROTICKET AND
TRUNC(B.FECHA_INTERACCION) BETWEEN TO_DATE(A.FECHA_INICIO, 'DD/MM/YYYY') AND TO_DATE(A.FECHA_FIN, 'DD/MM/YYYY')
WHERE A.FLAG_ETAPA='MENSAJE INI. CAMPAÑA' AND  A.FLAG_ACCION = 'OTRA RESPUESTA' AND B.ENTE_COMUNICADOR = 'PERSONA'
AND B.FECHA_INTERACCION > TO_DATE('01/07/2023', 'DD/MM/YYYY')
ORDER BY A.NUMEROTICKET, B.FECHA_INTERACCION, B.HORA_INTERACCION, B.ROWNUM_ID"""

        query2="""SELECT A.NUMEROTICKET, B.CONTENIDO_INTERACCION, B.ENTE_COMUNICADOR, B.FECHA_INTERACCION, B.HORA_INTERACCION, B.ROWNUM_ID,
A.FLAG_ETAPA, A.FLAG_ACCION, CAST(NULL AS VARCHAR2(100) ) AS  FRASE_ENCONTRADA ,
CAST(NULL AS NUMBER) AS TOXICIDAD, CAST(NULL AS VARCHAR2(100) ) AS  SENTIMIENTO 
FROM C26796_TEMP_UPGRADE_WSP_ETAPAS A
LEFT JOIN temp_wsp_det B
ON A.NUMEROTICKET= B.NUMEROTICKET AND
TRUNC(B.FECHA_INTERACCION) BETWEEN TO_DATE(A.FECHA_INICIO, 'DD/MM/YYYY') AND TO_DATE(A.FECHA_FIN, 'DD/MM/YYYY')
WHERE A.FLAG_ETAPA='ASIGNAR ASESOR' AND  A.FLAG_ACCION = 'NO VENTA' 
AND (B.ENTE_COMUNICADOR = 'PERSONA' OR B.ENTE_COMUNICADOR = 'ASESOR') 
AND B.CONTENIDO_INTERACCION NOT LIKE '%Click en%' AND  B.CONTENIDO_INTERACCION NOT LIKE  '%BOTON_%'
AND B.FECHA_INTERACCION > TO_DATE('01/07/2023', 'DD/MM/YYYY')
ORDER BY A.NUMEROTICKET, B.FECHA_INTERACCION, B.HORA_INTERACCION, B.ROWNUM_ID"""

        query3 = """SELECT A.NUMEROTICKET, B.CONTENIDO_INTERACCION, B.ENTE_COMUNICADOR, B.FECHA_INTERACCION, B.HORA_INTERACCION, B.ROWNUM_ID,
A.FLAG_ETAPA, A.FLAG_ACCION, CAST(NULL AS VARCHAR2(100) ) AS  FRASE_ENCONTRADA ,
CAST(NULL AS NUMBER) AS TOXICIDAD, CAST(NULL AS VARCHAR2(100) ) AS  SENTIMIENTO 
FROM C26796_TEMP_UPGRADE_WSP_ETAPAS A
LEFT JOIN temp_wsp_det B
ON A.NUMEROTICKET= B.NUMEROTICKET AND
TRUNC(B.FECHA_INTERACCION) BETWEEN TO_DATE(A.FECHA_INICIO, 'DD/MM/YYYY') AND TO_DATE(A.FECHA_FIN, 'DD/MM/YYYY')
WHERE A.FLAG_ETAPA='ASIGNAR ASESOR' AND  A.FLAG_ACCION = 'VENTA EXITOSA' 
AND (B.ENTE_COMUNICADOR = 'PERSONA' OR B.ENTE_COMUNICADOR = 'ASESOR') 
AND B.CONTENIDO_INTERACCION NOT LIKE '%Click en%' AND  B.CONTENIDO_INTERACCION NOT LIKE  '%BOTON_%'
AND B.FECHA_INTERACCION > TO_DATE('01/07/2023', 'DD/MM/YYYY')
ORDER BY A.NUMEROTICKET, B.FECHA_INTERACCION, B.HORA_INTERACCION, B.ROWNUM_ID"""



        
        self.WSP_UPG_procesar_conversaciones_oracle(query_TOTAL,frases_obj, "TODOS LOS WSP",4)
        
        #self.WSP_UPG_procesar_conversaciones_oracle(query1,frases_obj, "Mensaje ini. Otra respuesta",4)
        #self.WSP_UPG_procesar_conversaciones_oracle(query2,frases_obj, "Asignado asesor, no Venta ",8)
        #self.WSP_UPG_procesar_conversaciones_oracle(query3,frases_obj, "Asignado asesor, venta exitosa",8)


    def WSP_UPG_procesar_conversaciones_oracle(self, consulta_select, frases_obj, etiqueta ="", largo_frases=4):
        longitud_maxima_caracteres = 500
        conn = self.connectionDWO()  
        cursor = conn.cursor()
        ticket_actual=""
        conversacion =""
        general_text =""
 
        cursor.execute(consulta_select)
        resultados = cursor.fetchall()
 
        for resultado in resultados:
            numeroticket = resultado[0]
            contenido_interaccion = str(resultado[1])
            ente_comunicador = resultado[2]
            fecha_interaccion = resultado[3]
            hora_interaccion = resultado[4]
            rownum_id = int(resultado[5])
            flag_etapa = resultado[6]
            flag_accion = resultado[7] 
            frase_encontrada = ""
            frase_portout = ""
            frase_reno = ""
            frase_no_contact = ""
            frase_insatisfaccion = ""
            toxicidad = ""
            sentimiento = ""
            general_text= general_text + ". \n" + contenido_interaccion

            if ticket_actual!=numeroticket:
                palabras_ordenadas, lisuras_encontradas, preguntas_ordenadas, frases_encontradas_3, oraciones_encontradas = self.analisis_conversacion(conversacion, self.lisuras,largo_frases)
                #print("Palabras más usadas:", palabras_ordenadas)
                #print("Lisuras encontradas:", lisuras_encontradas)
                #print("Preguntas más frecuentes:", preguntas_ordenadas)


                try:
                    for orden, (palabra, frecuencia) in enumerate(palabras_ordenadas, start=1):
                        if frecuencia>1:
                            cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION (NUMEROTICKET, ORDEN, TIPO, VALOR,CANTIDAD,flag_etapa, flag_accion) VALUES (:1, :2, :3, :4, :5, :6, :7 )",
                            (ticket_actual, orden, 'PALABRA_FRECUENTE', f'{palabra}',frecuencia,flag_etapa, flag_accion))
 
                    for orden, (lisura, frecuencia) in enumerate(lisuras_encontradas.items(), start=1):
                        if frecuencia>0:
                            cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION (NUMEROTICKET, ORDEN, TIPO, VALOR,CANTIDAD ,flag_etapa, flag_accion) VALUES (:1, :2, :3, :4, :5, :6 , :7)",
                           (ticket_actual, orden, 'LISURA', f'{lisura}',frecuencia,flag_etapa, flag_accion))
 
                    for orden, (pregunta, frecuencia) in enumerate(preguntas_ordenadas, start=1):
                        if frecuencia>0:
                            cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION (NUMEROTICKET, ORDEN, TIPO, VALOR,CANTIDAD , flag_etapa, flag_accion) VALUES (:1, :2, :3, :4, :5, :6, :7)",
                               (ticket_actual, orden, 'PREGUNTA_FRECUENTE', f'{pregunta}',frecuencia,flag_etapa, flag_accion))
                        
                    for orden, (frase, frecuencia) in enumerate(frases_encontradas_3, start=1):
                        if frecuencia>1:
                            oracion  = ' '.join(frase)
                            cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION (NUMEROTICKET, ORDEN, TIPO, VALOR, CANTIDAD, flag_etapa, flag_accion) VALUES (:1, :2, :3, :4, :5, :6, :7)",
                            (ticket_actual, orden, 'FRASE_FRECUENTE', f'{oracion}',frecuencia,flag_etapa, flag_accion))
 
 
                    conn.commit()
                except cx_Oracle.DatabaseError as e:
                        print(f'Error al insertar en la base de datos: {e}')
                finally:
                        conversacion=""
                        #longitud maxima depende de la memoria de la computadora

 
            ticket_actual= numeroticket
            conversacion   = conversacion + " " +   contenido_interaccion   
            palabras = contenido_interaccion.split(" ")
            cantidad_grupos = round((len(contenido_interaccion) / longitud_maxima_caracteres) + 0.5)
            contador_palabras = 1
 
            if cantidad_grupos == 1:


            #analisis

                #toxicidad=str(self.detectar_toxicidad(contenido_interaccion)[0]['score'])
                #sentimiento = str(self.mostrar_analisis_sent(contenido_interaccion))
                found_frase_encontrada = self.find_similar_phrases(contenido_interaccion, frases_obj,2,0)
                found_frase_portout = self.find_similar_phrases(contenido_interaccion, frases_obj,2,4)
                found_frase_reno = self.find_similar_phrases(contenido_interaccion, frases_obj,2,5)
                found_frase_no_contact = self.find_similar_phrases(contenido_interaccion, frases_obj,2,6)
                found_frase_insatisfaccion = self.find_similar_phrases(contenido_interaccion, frases_obj,2,7)

                if found_frase_encontrada:frase_encontrada = str(found_frase_encontrada[0])
                if found_frase_portout:frase_portout = str(found_frase_portout[0])  #portout
                if found_frase_reno:frase_reno = str(found_frase_reno[0])  #reno
                if found_frase_no_contact:frase_no_contact = str(found_frase_no_contact[0])  # no contact
                if found_frase_insatisfaccion:frase_insatisfaccion = str(found_frase_insatisfaccion[0])  # no contact


                cursor.execute(f"INSERT INTO C26796_UPGRADE_WSP_CONVERSACION_DET (numeroticket, contenido_interaccion, ente_comunicador, fecha_interaccion, hora_interaccion, rownum_id, flag_etapa, flag_accion, frase_encontrada, frase_portout, frase_reno, frase_no_contact,frase_insatisfaccion, toxicidad, sentimiento) VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15 )", 
                               (numeroticket, contenido_interaccion,ente_comunicador, 
                    fecha_interaccion, hora_interaccion, rownum_id, flag_etapa, flag_accion, frase_encontrada, frase_portout, frase_reno, frase_no_contact,frase_insatisfaccion, toxicidad, sentimiento))
                conn.commit()
            else:
                for i in range(cantidad_grupos):
                    fragmento_conversacion = ""
                    for orden, palabra in enumerate(palabras, start=contador_palabras):
                        if len(fragmento_conversacion) >= longitud_maxima_caracteres:
                            break
                        fragmento_conversacion += " " + palabra
                        contador_palabras += 1
 
                #analisis
                    #toxicidad=str(self.detectar_toxicidad(contenido_interaccion)[0]['score'])
                    #sentimiento = str(self.mostrar_analisis_sent(contenido_interaccion))
                    found_frase_encontrada = self.find_similar_phrases(fragmento_conversacion, frases_obj,2,0)
                    if found_frase_encontrada:frase_encontrada = str(found_frase_encontrada[0])
                    cursor.execute(f"INSERT INTO C26796_UPGRADE_WSP_CONVERSACION_DET (numeroticket, contenido_interaccion, ente_comunicador, fecha_interaccion, hora_interaccion, rownum_id, flag_etapa, flag_accion, frase_encontrada, frase_portout, frase_reno, frase_no_contact,frase_insatisfaccion, toxicidad, sentimiento) VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15)", 
                    (numeroticket, fragmento_conversacion,ente_comunicador, 
                    fecha_interaccion, hora_interaccion, rownum_id, flag_etapa, flag_accion, frase_encontrada, frase_portout, frase_reno, frase_no_contact,frase_insatisfaccion, toxicidad, sentimiento))
                    conn.commit()
 
        conn.commit()


        palabras_ordenadas, lisuras_encontradas, preguntas_ordenadas, frases_encontradas, oraciones_encontradas = self.analisis_conversacion(general_text, self.lisuras,largo_frases)
        print("ANALISIS GLOBAL : " + etiqueta)

        #print(nltk.sent_tokenize(general_text.lower()))
        print("Palabras más usadas:", palabras_ordenadas)
        print("Lisuras encontradas:", lisuras_encontradas)
        print("Preguntas más frecuentes:", preguntas_ordenadas)
        print("Frases más frecuentes:", frases_encontradas)
        print("Oraciones más frecuentes:", oraciones_encontradas)
        




        try:
            toxicidad=""
            sentimiento = ""
            for orden, (palabra, frecuencia) in enumerate(palabras_ordenadas, start=1):
                if frecuencia>5:

                    cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION_GLOBAL (TIPO, VALOR, CANTIDAD, flag_etapa, flag_accion, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6, :7)",
                            ('PALABRA_FRECUENTE', f'{palabra}',frecuencia,flag_etapa, flag_accion,sentimiento, toxicidad))
                    
            for orden, (lisura, frecuencia) in enumerate(lisuras_encontradas.items(), start=1):
                
                cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION_GLOBAL (TIPO, VALOR, CANTIDAD, flag_etapa, flag_accion, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6, :7)",
                            ('LISURA', f'{lisura}',frecuencia,flag_etapa, flag_accion,sentimiento, toxicidad))
                    
            for orden, (pregunta, frecuencia) in enumerate(preguntas_ordenadas, start=1):
                
                cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION_GLOBAL (TIPO, VALOR, CANTIDAD, flag_etapa, flag_accion, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6, :7)",
                            ('PREGUNTA_FRECUENTE', f'{pregunta}',frecuencia,flag_etapa, flag_accion,sentimiento, toxicidad))  
                                      
            for orden, (frase, frecuencia) in enumerate(frases_encontradas, start=1):
                toxicidad=""
                sentimiento = ""
                if frecuencia>5:
                    oracion  = ' '.join(frase)
                    cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION_GLOBAL (TIPO, VALOR, CANTIDAD, flag_etapa, flag_accion, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6, :7)",
                            ('FRASE_FRECUENTE', f'{oracion}',frecuencia,flag_etapa, flag_accion, sentimiento, toxicidad))
        
            for orden, (oracion_c, frecuencia) in enumerate(oraciones_encontradas, start=1):
                toxicidad=""
                sentimiento = ""
                oracion  = ''.join(oracion_c)
                if frecuencia>20:
                    #toxicidad=str(self.detectar_toxicidad(oracion)[0]['score'])
                    #sentimiento = str(self.mostrar_analisis_sent(oracion))
                    pass
                cursor.execute("INSERT INTO C26796_UPGRADE_WSP_CONVERSACION_GLOBAL (TIPO, VALOR, CANTIDAD, flag_etapa, flag_accion, sentimiento, toxicidad) VALUES (:1, :2, :3, :4, :5, :6, :7)",
                            ('ORACION_FRECUENTE', f'{oracion}',frecuencia,flag_etapa, flag_accion, sentimiento, toxicidad))
        
            
            conn.commit()
        except cx_Oracle.DatabaseError as e:
                        print(f'Error al insertar en la base de datos: {e}')
            
        cursor.close()
        conn.close()




    def connectionDWO(self):
        dsn = cx_Oracle.makedsn('scan-dwo.claro.pe','1521',service_name='DWO')
        connection = cx_Oracle.connect(user=self.user, password=self.password, dsn = dsn)
        return connection
    


    #NO SE USA
    def insertar_en_base_de_datos(self,txt_folder, phrases_to_find):
        conn = self.connectionDWO()
        cursor = conn.cursor()
        frase_encontrada = ""
        frase_ofrecimiento = ""
        frase_venta = ""
        frase_cierre_venta = ""
        
 
        for filename in os.listdir(txt_folder):
            if filename.endswith(".txt"): #txt
                file_path = os.path.join(txt_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                frases = content.split("<.#div#>")
                for orden, frase in enumerate(frases, start=1):
                    frase_encontrada = ""
                    frase_ofrecimiento = ""
                    frase_venta = ""
                    frase_cierre_venta = ""

                    found_frase_encontrada = self.find_similar_phrases(frase, phrases_to_find,2,0)
                    if found_frase_encontrada:frase_encontrada = str(found_frase_encontrada[0])
                    
                    found_frase_ofrecimiento = self.find_similar_phrases(frase, phrases_to_find,2,1)
                    if found_frase_ofrecimiento:frase_ofrecimiento = str(found_frase_ofrecimiento[0])
                    
                    found_frase_venta = self.find_similar_phrases(frase, phrases_to_find,2,2)
                    if found_frase_venta:frase_venta = str(found_frase_venta[0])

                    found_frase_cierre_venta = self.find_similar_phrases(frase, phrases_to_find,2,3)
                    if found_frase_cierre_venta:frase_cierre_venta = str(found_frase_cierre_venta[0]) 
                    #print(frase)
                    
                    cursor.execute("INSERT INTO C26796_TEMP_CONVERSACION_CALL_UPGRADE (ID, ORDEN, MENSAJE, FRASE_ENCONTRADA,FRASE_OFRECIMIENTO,FRASE_VENTA,FRASE_CIERRE_VENTA) VALUES (:1, :2, :3, :4, :5, :6, :7)", 
                                   (filename, orden, frase, frase_encontrada, frase_ofrecimiento, frase_venta, frase_cierre_venta))  

        conn.commit()
        cursor.close()
        conn.close()
