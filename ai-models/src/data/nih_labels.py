import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import os

# Disease classes matching your config
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class NIHLabelsParser:
    """Parse NIH Chest X-ray labels into multi-hot vectors"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.meta_dir = self.data_dir / "meta"
        self.images_dir = self.data_dir / "images"
        
        # Create directories if they don't exist
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def create_dummy_data(self, num_samples=1000):
        """Create dummy data for testing when real NIH data isn't available"""
        
        print(f"📊 Creating {num_samples} dummy samples for testing...")
        
        # Create dummy CSV
        dummy_data = []
        
        for i in range(num_samples):
            # Random image name
            image_name = f"dummy_image_{i:05d}.png"
            
            # Random selection of diseases (can be multiple or none)
            selected_diseases = np.random.choice(
                DISEASE_CLASSES + ['No Finding'], 
                size=np.random.randint(0, 4),  # 0-3 diseases
                replace=False
            )
            
            # Join diseases with "|" 
            if len(selected_diseases) == 0 or 'No Finding' in selected_diseases:
                finding_labels = 'No Finding'
            else:
                finding_labels = '|'.join(selected_diseases)
            
            dummy_data.append({
                'Image Index': image_name,
                'Finding Labels': finding_labels,
                'Follow-up #': 0,
                'Patient ID': f'P{i:05d}',
                'Patient Age': np.random.randint(20, 90),
                'Patient Gender': np.random.choice(['M', 'F']),
                'View Position': np.random.choice(['PA', 'AP']),
                'OriginalImage[Width': np.random.randint(1024, 3000),
                'Height]': np.random.randint(1024, 3000),
                'OriginalImagePixelSpacing[x': 0.143,
                'y]': 0.143
            })
        
        # Save dummy CSV
        dummy_df = pd.DataFrame(dummy_data)
        csv_path = self.meta_dir / "Data_Entry_2017_dummy.csv"
        dummy_df.to_csv(csv_path, index=False)
        
        # Create dummy images (small black images)
        import cv2
        for i in range(min(num_samples, 100)):  # Only create first 100 images
            image_name = f"dummy_image_{i:05d}.png"
            image_path = self.images_dir / image_name
            
            # Create a 256x256 black image with some noise
            dummy_image = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), dummy_image)
        
        print(f"✅ Created dummy CSV: {csv_path}")
        print(f"✅ Created {min(num_samples, 100)} dummy images in: {self.images_dir}")
        
        return str(csv_path)
    
    def parse_labels(self, csv_path=None, max_samples=None):
        """
        Parse NIH labels CSV into image paths and multi-hot label vectors
        
        Args:
            csv_path: Path to CSV file (if None, looks for standard names)
            max_samples: Limit number of samples (for testing)
        
        Returns:
            Tuple of (image_paths, labels_array, metadata_df)
        """
        
        # Find CSV file
        if csv_path is None:
            # Try to find existing CSV files
            possible_csvs = [
                self.meta_dir / "Data_Entry_2017.csv",  # Real NIH data
                self.meta_dir / "Data_Entry_2017_dummy.csv"  # Dummy data
            ]
            
            csv_path = None
            for possible_csv in possible_csvs:
                if possible_csv.exists():
                    csv_path = possible_csv
                    break
            
            # If no CSV found, create dummy data
            if csv_path is None:
                print("⚠️  No NIH data found. Creating dummy data for testing...")
                csv_path = self.create_dummy_data(1000)
        
        print(f"📊 Loading labels from: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if max_samples:
            df = df.head(max_samples)
        
        print(f"📊 Loaded {len(df)} samples")
        
        # Parse labels into multi-hot vectors
        image_paths = []
        labels_list = []
        
        for idx, row in df.iterrows():
            image_name = row['Image Index']
            image_path = self.images_dir / image_name
            
            # Skip if image doesn't exist (for dummy data, only first 100 exist)
            if not image_path.exists():
                continue
                
            finding_labels = str(row['Finding Labels'])
            
            # Create multi-hot vector
            label_vector = np.zeros(len(DISEASE_CLASSES), dtype=np.float32)
            
            if finding_labels != 'No Finding' and pd.notna(finding_labels):
                # Split by "|" and find indices
                diseases = [disease.strip() for disease in finding_labels.split('|')]
                for disease in diseases:
                    if disease in DISEASE_CLASSES:
                        idx = DISEASE_CLASSES.index(disease)
                        label_vector[idx] = 1.0
            
            image_paths.append(str(image_path))
            labels_list.append(label_vector)
        
        labels_array = np.array(labels_list)
        
        print(f"✅ Processed {len(image_paths)} valid samples")
        print(f"📊 Label distribution (positive samples per class):")
        
        for i, disease in enumerate(DISEASE_CLASSES):
            count = np.sum(labels_array[:, i])
            print(f"   {disease}: {int(count)} ({count/len(labels_array)*100:.1f}%)")
        
        return image_paths, labels_array, df

if __name__ == "__main__":
    # Test the parser
    parser = NIHLabelsParser()
    
    # Parse labels
    image_paths, labels, metadata = parser.parse_labels(max_samples=500)
    
    print(f"\n🎯 Sample Results:")
    print(f"Total samples: {len(image_paths)}")
    print(f"Labels shape: {labels.shape}")
    print(f"First image: {image_paths[0]}")
    print(f"First label: {labels[0]}")
    print(f"Diseases in first sample: {[DISEASE_CLASSES[i] for i in range(len(DISEASE_CLASSES)) if labels[0][i] == 1]}")