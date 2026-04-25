import cv2

class CameraManager:
    """
    Gerenciador da câmera do dispositivo.
    Garante que os recursos de hardware sejam liberados após o uso.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.captura = None

    def __enter__(self):
        """
        Inicializa a câmera quando o bloco 'with' começa.
        """
        self.captura = cv2.VideoCapture(self.camera_index)

        if not self.captura.isOpened():
            raise RuntimeError(f"Erro de Segurança/Hardware: Não foi possível acessar a câmera {self.camera_index}.")
        return self.captura

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Garante que a câmera seja desligada quanto o bloco 'with' termina ou dá erro.
        """
        if self.captura is not None:
            self.captura.release()
        cv2.destroyAllWindows()
        print("[INFO] Câmera desligada com segurança e recursos liberados.")
