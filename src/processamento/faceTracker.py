import cv2

class RastreamentoFacial:
    """
    Módulo de Rastreamento usando OpenCV nativo.
    Abandona o MediaPipe devido a incompatibilidades críticas e foca na extração rápida da face.
    """
    def __init__(self, max_rostos=5):
        # Carrega o modelo de detecção facial Haar Cascade que já vem embutido dentro do OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector_rosto = cv2.CascadeClassifier(cascade_path)

    def processar_frame(self, frame):
        """Converte a imagem para tons de cinza (mais leve) e localiza os rostos."""
        frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # O modelo retorna as coordenadas: x, y, largura (w) e altura (h)
        rostos = self.detector_rosto.detectMultiScale(
            frame_cinza,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50) # Ignora ruídos muito pequenos
        )
        return rostos

    def desenhar_landmarks(self, frame, rostos):
        """Desenha um retângulo ao redor de cada rosto detectado para feedback visual."""
        for (x, y, w, h) in rostos:
            # Desenha a Caixa Delimitadora (Bounding Box) em verde
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Deixa um marcador de texto preparado para a nossa futura IA de Emoções
            cv2.putText(frame, "Analisando Emocao...", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        return frame