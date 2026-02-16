import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import numpy as np
import json

class Classifier:
    def __init__(self, num_classes=1000, model_path=None):
        """Initialize classifier using ImageNet pretrained model"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Use pretrained ResNet50 with full ImageNet classes
        self.model = models.resnet50(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load ImageNet class names
        self.class_names = self._load_imagenet_classes()
        
        # Define human and animal class ranges
        self.human_keywords = ['person', 'man', 'woman', 'boy', 'girl', 'child', 'people']
        self.animal_start_idx = 151  # Animals start from class 151 in ImageNet
    
    def _load_imagenet_classes(self):
        """Load ImageNet class names"""
        # Simplified ImageNet classes (key animal classes)
        classes = {
            0: 'person', 151: 'chihuahua', 152: 'japanese_spaniel', 153: 'maltese_dog',
            154: 'pekinese', 155: 'shih-tzu', 156: 'blenheim_spaniel', 157: 'papillon',
            158: 'toy_terrier', 159: 'rhodesian_ridgeback', 160: 'afghan_hound',
            161: 'basset', 162: 'beagle', 163: 'bloodhound', 164: 'bluetick',
            165: 'black-and-tan_coonhound', 166: 'walker_hound', 167: 'english_foxhound',
            168: 'redbone', 169: 'borzoi', 170: 'irish_wolfhound', 171: 'italian_greyhound',
            172: 'whippet', 173: 'ibizan_hound', 174: 'norwegian_elkhound', 175: 'otterhound',
            176: 'saluki', 177: 'scottish_deerhound', 178: 'weimaraner',
            179: 'staffordshire_bullterrier', 180: 'american_staffordshire_terrier',
            181: 'bedlington_terrier', 182: 'border_terrier', 183: 'kerry_blue_terrier',
            184: 'irish_terrier', 185: 'norfolk_terrier', 186: 'norwich_terrier',
            187: 'yorkshire_terrier', 188: 'wire-haired_fox_terrier', 189: 'lakeland_terrier',
            190: 'sealyham_terrier', 191: 'airedale', 192: 'cairn', 193: 'australian_terrier',
            194: 'dandie_dinmont', 195: 'boston_bull', 196: 'miniature_schnauzer',
            197: 'giant_schnauzer', 198: 'standard_schnauzer', 199: 'scotch_terrier',
            200: 'tibetan_terrier', 201: 'silky_terrier', 202: 'soft-coated_wheaten_terrier',
            203: 'west_highland_white_terrier', 204: 'lhasa', 205: 'flat-coated_retriever',
            206: 'curly-coated_retriever', 207: 'golden_retriever', 208: 'labrador_retriever',
            209: 'chesapeake_bay_retriever', 210: 'german_short-haired_pointer',
            211: 'vizsla', 212: 'english_setter', 213: 'irish_setter', 214: 'gordon_setter',
            215: 'brittany_spaniel', 216: 'clumber', 217: 'english_springer',
            218: 'welsh_springer_spaniel', 219: 'cocker_spaniel', 220: 'sussex_spaniel',
            221: 'irish_water_spaniel', 222: 'kuvasz', 223: 'schipperke', 224: 'groenendael',
            225: 'malinois', 226: 'briard', 227: 'kelpie', 228: 'komondor',
            229: 'old_english_sheepdog', 230: 'shetland_sheepdog', 231: 'collie',
            232: 'border_collie', 233: 'bouvier_des_flandres', 234: 'rottweiler',
            235: 'german_shepherd', 236: 'doberman', 237: 'miniature_pinscher',
            238: 'greater_swiss_mountain_dog', 239: 'bernese_mountain_dog',
            240: 'appenzeller', 241: 'entlebucher', 242: 'boxer', 243: 'bull_mastiff',
            244: 'tibetan_mastiff', 245: 'french_bulldog', 246: 'great_dane',
            247: 'saint_bernard', 248: 'eskimo_dog', 249: 'malamute', 250: 'siberian_husky',
            251: 'affenpinscher', 252: 'basenji', 253: 'pug', 254: 'leonberg',
            255: 'newfoundland', 256: 'great_pyrenees', 257: 'samoyed',
            258: 'pomeranian', 259: 'chow', 260: 'keeshond', 261: 'brabancon_griffon',
            262: 'pembroke', 263: 'cardigan', 264: 'toy_poodle', 265: 'miniature_poodle',
            266: 'standard_poodle', 267: 'mexican_hairless', 268: 'dingo', 269: 'dhole',
            270: 'african_hunting_dog', 
            # Big cats and wild animals
            281: 'tabby_cat', 282: 'tiger_cat', 283: 'persian_cat', 284: 'siamese_cat',
            285: 'egyptian_cat', 286: 'cougar', 287: 'lynx', 288: 'leopard', 289: 'snow_leopard',
            290: 'jaguar', 291: 'lion', 292: 'tiger', 293: 'cheetah', 294: 'brown_bear',
            295: 'american_black_bear', 296: 'ice_bear', 297: 'sloth_bear', 298: 'mongoose',
            299: 'meerkat', 300: 'tiger_beetle', 301: 'ladybug', 302: 'ground_beetle',
            # More animals
            330: 'hamster', 331: 'porcupine', 332: 'fox_squirrel', 333: 'marmot',
            334: 'beaver', 335: 'guinea_pig', 336: 'sorrel', 337: 'zebra', 338: 'hog',
            339: 'wild_boar', 340: 'warthog', 341: 'hippopotamus', 342: 'ox',
            343: 'water_buffalo', 344: 'bison', 345: 'ram', 346: 'bighorn', 347: 'ibex',
            348: 'hartebeest', 349: 'impala', 350: 'gazelle', 351: 'arabian_camel',
            352: 'llama', 353: 'weasel', 354: 'mink', 355: 'polecat', 356: 'black-footed_ferret',
            357: 'otter', 358: 'skunk', 359: 'badger', 360: 'armadillo',
            361: 'three-toed_sloth', 362: 'orangutan', 363: 'gorilla', 364: 'chimpanzee',
            365: 'gibbon', 366: 'siamang', 367: 'guenon', 368: 'patas', 369: 'baboon',
            370: 'macaque', 371: 'langur', 372: 'colobus', 373: 'proboscis_monkey',
            374: 'marmoset', 375: 'capuchin', 376: 'howler_monkey', 377: 'titi',
            378: 'spider_monkey', 379: 'squirrel_monkey', 380: 'madagascar_cat',
            381: 'indri', 382: 'indian_elephant', 383: 'african_elephant', 384: 'lesser_panda',
            385: 'giant_panda', 386: 'barracouta', 387: 'eel', 388: 'coho',
            # Birds
            7: 'cock', 8: 'hen', 9: 'ostrich', 10: 'brambling', 11: 'goldfinch',
            12: 'house_finch', 13: 'junco', 14: 'indigo_bunting', 15: 'robin',
            16: 'bulbul', 17: 'jay', 18: 'magpie', 19: 'chickadee', 20: 'water_ouzel',
            21: 'kite', 22: 'bald_eagle', 23: 'vulture', 24: 'great_grey_owl',
            80: 'black_grouse', 81: 'ptarmigan', 82: 'ruffed_grouse', 83: 'prairie_chicken',
            84: 'peacock', 85: 'quail', 86: 'partridge', 87: 'african_grey',
            88: 'macaw', 89: 'sulphur-crested_cockatoo', 90: 'lorikeet',
            91: 'coucal', 92: 'bee_eater', 93: 'hornbill', 94: 'hummingbird',
            95: 'jacamar', 96: 'toucan', 97: 'drake', 98: 'red-breasted_merganser',
            99: 'goose', 100: 'black_swan', 101: 'white_stork', 102: 'black_stork',
            127: 'flamingo', 128: 'little_blue_heron', 129: 'american_egret',
            130: 'bittern', 131: 'crane', 132: 'limpkin', 133: 'european_gallinule',
            134: 'american_coot', 135: 'bustard', 136: 'ruddy_turnstone',
            137: 'red-backed_sandpiper', 138: 'redshank', 139: 'dowitcher',
            140: 'oystercatcher', 141: 'pelican', 142: 'king_penguin',
            143: 'albatross', 144: 'grey_whale', 145: 'killer_whale', 146: 'dugong',
            147: 'sea_lion',
        }
        return classes
    
    def classify(self, image):
        """Classify cropped object and return only category (animal/human)"""
        if image is None or image.size == 0:
            return 'unknown', 0.0
            
        try:
            # Convert to PIL
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
            else:
                img = image
            
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top 5 predictions
                top5_prob, top5_classes = torch.topk(probs, 5)
                
                # Check all top 5 predictions
                for i in range(5):
                    class_idx = top5_classes[0][i].item()
                    confidence = top5_prob[0][i].item()
                    
                    # Check if it's an animal (class >= 151)
                    if class_idx >= 151 and class_idx < 400:
                        return 'animal', confidence
                
                # If no animal found in top 5, assume it's human or unknown
                # Check if image has human-like features (vertical aspect ratio, skin tones)
                img_array = np.array(img)
                height, width = img_array.shape[:2]
                aspect_ratio = height / width if width > 0 else 1
                
                # Humans typically have vertical aspect ratio (taller than wide)
                if aspect_ratio > 1.2:
                    return 'human', top5_prob[0][0].item()
                
                # Otherwise unknown
                return 'unknown', top5_prob[0][0].item()
                        
        except Exception as e:
            print(f"Classification error: {e}")
            return 'unknown', 0.0
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        """Train classifier"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
        
        return self.model
