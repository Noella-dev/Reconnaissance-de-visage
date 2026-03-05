import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import random

# ============================================================================
# 1. DÉTECTEUR DE VISAGES avec OpenCV
# ============================================================================
class FaceDetector:
    """Détecte les visages dans les images"""
    
    def __init__(self):
        # Chargeur de cascade Haar pour détection de visages
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("❌ Erreur: Impossible de charger le détecteur de visages!")
            self._download_cascade()
    
    def _download_cascade(self):
        """Télécharge le cascade classifier si absent"""
        print("📥 Tentative de téléchargement du détecteur...")
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
            self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            print("✅ Détecteur téléchargé")
        except:
            print("❌ Échec du téléchargement")
    
    def detect_faces(self, image):
        """Détecte tous les visages dans une image"""
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def extract_face(self, image, face_coords, target_size=(100, 100)):
        """Extrait et redimensionne un visage"""
        x, y, w, h = face_coords
        
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image.copy()
        
        face_img = img_np[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_img).convert('RGB')
        face_pil = face_pil.resize(target_size)
        
        return face_pil
    
    def draw_faces(self, image, faces, labels=None):
        """Dessine les rectangles et labels sur l'image"""
        result = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            if labels and i < len(labels):
                label = labels[i]
                if "MANOU" in label.upper():
                    color = (0, 255, 0)
                elif "INCONNU" in label.upper() or "PAS MANOU" in label.upper():
                    color = (0, 0, 255)
                else:
                    color = (255, 165, 0)
                
                cv2.rectangle(result, (x, y), (x+w, y+h), color, 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(result, label, (x, y-10), font, 0.7, color, 2)
            else:
                cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return result

# ============================================================================
# 2. MODÈLE DE RECONNAISSANCE AMÉLIORÉ
# ============================================================================
class ManouRecognizer(nn.Module):
    """Modèle CNN amélioré pour reconnaître Manou"""
    
    def __init__(self):
        super(ManouRecognizer, self).__init__()
        
        # Architecture CNN plus profonde
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)  # Dropout réduit
        
        # Pour images 100x100 -> après 3 pools: 12x12
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ============================================================================
# 3. DATASET PERSONNALISÉ AMÉLIORÉ
# ============================================================================
class ManouDataset(Dataset):
    """Dataset amélioré pour l'entraînement"""
    
    def __init__(self, manou_dir, autres_dir=None, transform=None, augment=True):
        self.transform = transform
        self.images = []
        self.labels = []
        
        print("📂 Chargement des images...")
        
        # Charger les photos de Manou
        if os.path.exists(manou_dir):
            manou_images = glob.glob(os.path.join(manou_dir, "*.jpg")) + \
                          glob.glob(os.path.join(manou_dir, "*.png")) + \
                          glob.glob(os.path.join(manou_dir, "*.jpeg"))
            
            # Exclure les résultats précédents
            manou_images = [img for img in manou_images if '_result' not in img]
            
            print(f"  👤 Photos de Manou trouvées: {len(manou_images)}")
            
            for img_path in manou_images:
                try:
                    img = Image.open(img_path).convert('RGB')
                    self.images.append(img.copy())
                    self.labels.append(1)
                    
                    # Augmentation MODÉRÉE
                    if augment and len(manou_images) < 15:
                        # Miroir horizontal seulement
                        mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
                        self.images.append(mirrored.copy())
                        self.labels.append(1)
                        
                except Exception as e:
                    print(f"    ❌ Erreur avec {img_path}: {e}")
        
        # Charger les photos d'autres personnes
        autres_count = 0
        if autres_dir and os.path.exists(autres_dir):
            autres_images = glob.glob(os.path.join(autres_dir, "*.jpg")) + \
                           glob.glob(os.path.join(autres_dir, "*.png")) + \
                           glob.glob(os.path.join(autres_dir, "*.jpeg"))
            
            print(f"  👥 Photos d'autres trouvées: {len(autres_images)}")
            
            for img_path in autres_images:
                try:
                    img = Image.open(img_path).convert('RGB')
                    self.images.append(img.copy())
                    self.labels.append(0)
                    autres_count += 1
                except:
                    pass
        
        # Générer des visages synthétiques SEULEMENT si vraiment nécessaire
        num_manou = sum(self.labels)
        num_autres = len(self.labels) - num_manou
        
        if num_autres < num_manou // 2:
            needed = num_manou - num_autres
            print(f"  🎭 Génération de {needed} visages synthétiques...")
            for _ in range(needed):
                fake_face = self._generate_fake_face()
                self.images.append(fake_face)
                self.labels.append(0)
        
        print(f"✅ Dataset créé: {len(self.images)} images total")
        print(f"   - Manou: {sum(self.labels)} images")
        print(f"   - Autres: {len(self.labels) - sum(self.labels)} images")
    
    def _generate_fake_face(self):
        """Génère un visage synthétique plus réaliste"""
        from PIL import ImageDraw, ImageFilter
        
        img = Image.new('RGB', (100, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Fond coloré aléatoire
        bg_color = (random.randint(180, 230), 
                   random.randint(180, 230), 
                   random.randint(180, 230))
        draw.rectangle([0, 0, 100, 100], fill=bg_color)
        
        # Visage ovale
        face_color = (random.randint(200, 255), 
                     random.randint(150, 200), 
                     random.randint(140, 190))
        draw.ellipse([15, 15, 85, 85], fill=face_color, outline='black', width=2)
        
        # Yeux
        eye_y = random.randint(35, 45)
        draw.ellipse([30, eye_y-5, 40, eye_y+5], fill='white', outline='black')
        draw.ellipse([60, eye_y-5, 70, eye_y+5], fill='white', outline='black')
        draw.ellipse([34, eye_y-2, 36, eye_y+2], fill='black')
        draw.ellipse([64, eye_y-2, 66, eye_y+2], fill='black')
        
        # Nez
        draw.line([50, eye_y+10, 50, eye_y+20], fill='gray', width=1)
        
        # Bouche
        mouth_y = random.randint(60, 70)
        draw.arc([35, mouth_y-5, 65, mouth_y+10], 0, 180, fill='red', width=2)
        
        # Flou léger pour rendre plus réaliste
        img = img.filter(ImageFilter.SMOOTH)
        
        return img
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# 4. SYSTÈME COMPLET AMÉLIORÉ
# ============================================================================
class ManouRecognitionSystem:
    """Système complet amélioré"""
    
    def __init__(self, model_path=None):
        print("="*60)
        print("   SYSTÈME 'EST-CE MANOU?' - VERSION AMÉLIORÉE")
        print("="*60)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Device: {self.device}")
        
        self.detector = FaceDetector()
        
        # Transformations SIMPLIFIÉES pour éviter trop de variations
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.model = ManouRecognizer().to(self.device)
        
        # Seuil de confiance AJUSTÉ
        self.confidence_threshold = 0.95
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif os.path.exists('modele_manou.pth'):
            self.load_model('modele_manou.pth')
            print("✅ Modèle chargé depuis modele_manou.pth")
        else:
            print("⚠️  Aucun modèle trouvé. Entraînez d'abord le système.")
    
    def train(self, manou_dir='photos_manou', autres_dir='photos_autres', epochs=20):
        """Entraîne le modèle avec paramètres améliorés"""
        print("\n🎯 DÉBUT DE L'ENTRAÎNEMENT AMÉLIORÉ")
        print(f"   - Dossier Manou: {manou_dir}")
        print(f"   - Dossier autres: {autres_dir}")
        print(f"   - Epochs: {epochs}")
        
        dataset = ManouDataset(manou_dir, autres_dir, self.transform, augment=True)
        
        if len(dataset) == 0:
            print("❌ Aucune image trouvée!")
            return False
        
        # Split 80/20
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Optimizer avec learning rate plus petit
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        
        print(f"\n📊 Entraînement sur {len(train_dataset)} images, validation sur {len(val_dataset)}")
        
        for epoch in range(epochs):
            # Entraînement
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Scheduler
            scheduler.step(val_loss)
            
            # Historique
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(100 * train_correct / train_total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Affichage
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train: {history['train_acc'][-1]:5.1f}% | "
                  f"Val: {val_acc:5.1f}% | "
                  f"Loss: {history['train_loss'][-1]:.4f}")
            
            # Sauvegarder le meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('modele_manou.pth')
                print(f"   💾 Meilleur modèle sauvegardé (Val: {val_acc:.1f}%)")
        
        self._plot_training_history(history)
        
        print(f"\n✅ Entraînement terminé!")
        print(f"   Meilleure précision validation: {best_val_acc:.1f}%")
        return True
    
    def _validate(self, val_loader, criterion):
        """Validation du modèle"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return val_loss / len(val_loader), 100 * correct / total
    
    def _plot_training_history(self, history):
        """Affiche les courbes d'apprentissage"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history['train_loss'], 'b-', label='Train', linewidth=2)
        ax1.plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Courbe de Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history['train_acc'], 'b-', label='Train', linewidth=2)
        ax2.plot(history['val_acc'], 'r-', label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Courbe de Précision')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('entrainement_manou.png', dpi=150)
        plt.show()
    
    def save_model(self, path='modele_manou.pth'):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'confidence_threshold': self.confidence_threshold
        }, path)
    
    def load_model(self, path):
        """Charge un modèle sauvegardé"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'confidence_threshold' in checkpoint:
            self.confidence_threshold = checkpoint['confidence_threshold']
        self.model.eval()
        print(f"📂 Modèle chargé: {path}")
    
    def is_manou(self, image):
        """Détermine si une image contient Manou"""
        if isinstance(image, str):
            if not os.path.exists(image):
                print(f"❌ Image non trouvée: {image}")
                return False, 0.0, []
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        faces = self.detector.detect_faces(image_np)
        
        if len(faces) == 0:
            print("👀 Aucun visage détecté")
            return False, 0.0, []
        
        print(f"👥 {len(faces)} visage(s) détecté(s)")
        
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            face_img = self.detector.extract_face(image_np, (x, y, w, h))
            face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(face_tensor)
                probabilities = F.softmax(output, dim=1)
                
                # Probabilités pour chaque classe
                prob_not_manou = probabilities[0][0].item()
                prob_is_manou = probabilities[0][1].item()
                
                # Décision basée sur la classe avec la plus haute probabilité
                if prob_is_manou > 0.95:  # Seuil de certitude très haut
                    is_manou = True
                    conf_value = prob_is_manou
                    label = f"✅ C'EST MANOU! ({conf_value:.1%})"
                else:
                    is_manou = False
                    conf_value = prob_not_manou
                    label = f"❌ INCONNU ({prob_is_manou:.1%})"
                
                results.append({
                    'face_coords': (x, y, w, h),
                    'is_manou': is_manou,
                    'confidence': conf_value,
                    'label': label,
                    'prob_manou': prob_is_manou,
                    'prob_not_manou': prob_not_manou
                })
                
                print(f"  Visage {i+1}: {label}")
                print(f"    Prob Manou: {prob_is_manou:.1%} | Prob Pas Manou: {prob_not_manou:.1%}")
        
        any_manou = any(r['is_manou'] for r in results)
        max_confidence = max(r['confidence'] for r in results) if results else 0.0
        
        return any_manou, max_confidence, results
    
    def analyze_image(self, image_path, display=True):
        """Analyse une image complète"""
        print(f"\n🔍 Analyse de: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        is_manou, confidence, face_results = self.is_manou(image_np)
        
        labels = [r['label'] for r in face_results]
        face_coords = [r['face_coords'] for r in face_results]
        
        if display and len(face_coords) > 0:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            annotated = self.detector.draw_faces(image_bgr, face_coords, labels)
            
            global_msg = "✅ C'EST MANOU!" if is_manou else "❌ CE N'EST PAS MANOU"
            cv2.putText(annotated, global_msg, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0) if is_manou else (0, 0, 255), 3)
            
            cv2.imshow("Résultat - Est-ce Manou?", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            result_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
            cv2.imwrite(result_path, annotated)
            print(f"💾 Résultat sauvegardé: {result_path}")
        
        print(f"\n📋 RÉSUMÉ:")
        print(f"   Image: {os.path.basename(image_path)}")
        print(f"   Visages détectés: {len(face_results)}")
        print(f"   Conclusion: {'✅ C\'EST MANOU!' if is_manou else '❌ CE N\'EST PAS MANOU'}")
        
        return {
            'is_manou': is_manou,
            'confidence': confidence,
            'faces_detected': len(face_results),
            'face_results': face_results
        }
    
    def real_time_detection(self, camera_id=0):
        print("\n🎥 DÉTECTION EN TEMPS RÉEL")
        print("📌 Appuyez sur 'q' pour quitter")
        
        cap = cv2.VideoCapture(camera_id)
        # Optimisation de la résolution pour éviter les lags
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra")
            return
        
        # Créer une fenêtre nommée fixe pour éviter les ouvertures multiples
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # On ne traite l'image que si nécessaire pour la fluidité
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            is_manou, confidence, face_results = self.is_manou(rgb_frame)
            
            display_frame = frame.copy()
            if face_results:
                face_coords = [r['face_coords'] for r in face_results]
                labels = [r['label'] for r in face_results]
                display_frame = self.detector.draw_faces(display_frame, face_coords, labels)
            
            # Affichage texte
            color = (0, 255, 0) if is_manou else (0, 0, 255)
            msg = "MANOU" if is_manou else "INCONNU"
            cv2.putText(display_frame, f"{msg} ({confidence:.1%})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Detection", display_frame)
            
            # IMPORTANT : Attend 1ms et vérifie si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyWindow("Detection") # Ferme uniquement cette fenêtre
# ============================================================================
# 5. INTERFACE UTILISATEUR
# ============================================================================
def cleanup():
    """Nettoie toutes les fenêtres OpenCV"""
    cv2.destroyAllWindows()
    print("🧹 Fenêtres fermées")

def setup_directories():
    """Crée les dossiers nécessaires"""
    directories = ['photos_manou', 'photos_autres', 'results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Dossier créé: {directory}/")
        else:
            images = glob.glob(os.path.join(directory, "*.jpg")) + \
                    glob.glob(os.path.join(directory, "*.png")) + \
                    glob.glob(os.path.join(directory, "*.jpeg"))
            images = [img for img in images if '_result' not in img]
            print(f"📂 Dossier {directory}/: {len(images)} images")
    
    print("\n📋 STRUCTURE REQUISE:")
    print("photos_manou/     ← VOS photos (.jpg, .png)")
    print("photos_autres/    ← Photos d'autres")
    print("results/          ← Résultats")
    print("="*60)

def main_menu():
    """Menu principal"""
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("    SYSTÈME 'EST-CE MANOU?' - VERSION AMÉLIORÉE")
    print("="*60)
    
    setup_directories()
    system = ManouRecognitionSystem()
    
    while True:
        print("\n" + "="*40)
        print("MENU PRINCIPAL")
        print("="*40)
        print("1. 🎯 Entraîner le modèle (RECOMMENCEZ ICI)")
        print("2. 🔍 Analyser une image")
        print("3. 🎥 Détection temps réel (webcam)")
        print("4. ⚙️  Tester le détecteur")
        print("5. 📊 Voir les courbes")
        print("6. 🧹 Fermer les fenêtres")
        print("7. 🚪 Quitter")
        
        choice = input("\nVotre choix (1-7): ").strip()
        
        if choice == '1':
            print("\n📝 ENTRAÎNEMENT AMÉLIORÉ")
            
            manou_dir = 'photos_manou'
            autres_dir = 'photos_autres'
            
            manou_photos = glob.glob(os.path.join(manou_dir, "*.jpg")) + \
                          glob.glob(os.path.join(manou_dir, "*.png")) + \
                          glob.glob(os.path.join(manou_dir, "*.jpeg"))
            manou_photos = [p for p in manou_photos if '_result' not in p]
            
            if len(manou_photos) == 0:
                print(f"\n⚠️  Aucune photo dans {manou_dir}/")
                print(f"Ajoutez des photos et réessayez!")
                continue
            
            print(f"\n✅ {len(manou_photos)} photos trouvées")
            
            epochs = input("Nombre d'epochs [20]: ").strip()
            epochs = int(epochs) if epochs else 20
            
            print(f"\n🚀 Entraînement pour {epochs} epochs...")
            system.train(manou_dir=manou_dir, autres_dir=autres_dir, epochs=epochs)
        
        elif choice == '2':
            print("\n🖼️  ANALYSE D'IMAGE")
            
            all_images = []
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                all_images.extend(glob.glob(os.path.join('photos_manou', ext)))
                all_images.extend(glob.glob(os.path.join('photos_autres', ext)))
            all_images = [img for img in all_images if '_result' not in img]
            
            if all_images:
                print("\n📷 Images disponibles:")
                for i, img_path in enumerate(all_images[:10]):
                    print(f"  {i+1}. {os.path.basename(img_path)}")
                if len(all_images) > 10:
                    print(f"  ... et {len(all_images)-10} autres")
            
            image_path = input("\nChemin (ou numéro): ").strip()
            
            if image_path.isdigit():
                idx = int(image_path) - 1
                if 0 <= idx < len(all_images):
                    image_path = all_images[idx]
                else:
                    print("❌ Numéro invalide!")
                    continue
            
            if image_path and os.path.exists(image_path):
                system.analyze_image(image_path, display=True)
            else:
                print("❌ Fichier non trouvé!")
        
        elif choice == '3':
            print("\n🎥 MODE TEMPS RÉEL")
            camera_id = input("ID caméra [0]: ").strip()
            camera_id = int(camera_id) if camera_id else 0
            system.real_time_detection(camera_id=camera_id)
        
        elif choice == '4':
            print("\n⚙️  TEST DÉTECTEUR")
            test_path = input("Image (vide pour webcam): ").strip()
            
            detector = FaceDetector()
            
            if test_path and os.path.exists(test_path):
                img = cv2.imread(test_path)
                faces = detector.detect_faces(img)
                print(f"👥 {len(faces)} visage(s)")
                result = detector.draw_faces(img, faces)
                cv2.imshow("Test", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Webcam inaccessible")
                else:
                    print("📹 Webcam - 'q' pour quitter")
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        faces = detector.detect_faces(frame)
                        result = detector.draw_faces(frame, faces)
                        cv2.putText(result, f"Visages: {len(faces)}", (20, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Test Webcam", result)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    cap.release()
                    cv2.destroyAllWindows()
        
        elif choice == '5':
            if os.path.exists('entrainement_manou.png'):
                img = plt.imread('entrainement_manou.png')
                plt.figure(figsize=(12, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            else:
                print("❌ Aucune courbe trouvée")
        
        elif choice == '6':
            cleanup()
        
        elif choice == '7':
            cleanup()
            print("\n👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide!")

# ============================================================================
# EXÉCUTION
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("Système 'Est-ce Manou?' - VERSION AMÉLIORÉE")
    print("="*60)
    print("CORRECTIONS:")
    print("- Meilleur modèle CNN avec BatchNorm")
    print("- Augmentation de données modérée")
    print("- Affichage des probabilités détaillées")
    print("- Seuil de confiance ajusté à 70%")
    print("="*60)
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption")
        cleanup()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        cleanup()
        raise