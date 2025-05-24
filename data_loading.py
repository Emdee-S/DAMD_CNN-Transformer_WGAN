# load the three datasets 128x128 with labels 
import numpy as np 
import gdown
import zipfile 


def download_data():
    
    # download and unzip the clean data 
    file_id = '1Gf532J0NEthwhg0k20oC7lJforGkmklx'
    url = f'https://drive.google.com/uc?id={file_id}'   
    gdown.download(url, 'data_zipped', quiet=False) 
    with zipfile.ZipFile('data_zipped', 'r') as zipped_ref:
            zipped_ref.extractall('data_unzipped')

def get_data_unbalanced():
    # load data
    poland_images = np.load("/content/data_unzipped/Datasets/BrEaST Lesions - Breast Ultrasound 128x128/Poland_Medical_Centers_US_Breast_Images.npy")
    poland_labels = np.load("/content/data_unzipped/Datasets/BrEaST Lesions - Breast Ultrasound 128x128/Poland_Medical_Centers_US_Breast_Labels.npy")

    china_images = np.load("/content/data_unzipped/Datasets/Chinese Hospitals - Kidney Ultrasound 128x128/Chinese Hospitals_Kidney_US_Images.npy")
    china_labels = np.load("/content/data_unzipped/Datasets/Chinese Hospitals - Kidney Ultrasound 128x128/Chinese Hospitals_Kidney_US_Labels.npy")

    egypt_images = np.load("/content/data_unzipped/Datasets/Egypt Hospital - Breast Ultrasound/Egypt Hospital 128x128/Egypt Hospital_US_Breast_Images.npy")
    egypt_labels = np.load("/content/data_unzipped/Datasets/Egypt Hospital - Breast Ultrasound/Egypt Hospital 128x128/Egypt Hospital_US_Breast_Labels.npy")
    
    def shuffle_data(images, labels):
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        return images[indices], labels[indices]
    
    # Shuffle each dataset
    poland_images, poland_labels = shuffle_data(poland_images, poland_labels)
    china_images, china_labels = shuffle_data(china_images, china_labels)
    egypt_images, egypt_labels = shuffle_data(egypt_images, egypt_labels)
    
    # check
    print(f'Poland: images {poland_images.shape}, labels {poland_labels.shape}')
    print(f'China: images {china_images.shape}, labels {china_labels.shape}')
    print(f'Egypt: images {egypt_images.shape}, labels {egypt_labels.shape}')
    
    # Poland
    values, counts = np.unique(poland_labels, return_counts=True)
    print("0 for malignant, 1 for benign,")
    print("Poland label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}\n")
    
    # China
    values, counts = np.unique(china_labels, return_counts=True)
    print("China label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}\n")
    
    # Egypt
    values, counts = np.unique(egypt_labels, return_counts=True)
    print("Egypt label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}") 
    
    return (poland_images, poland_labels), (china_images, china_labels), (egypt_images, egypt_labels)
    
def get_data_balanced_shuffled():
    # Load unbalanced data
    (poland_images, poland_labels), (china_images, china_labels), (egypt_images, egypt_labels) = get_data_unbalanced()

    # called below
    def balance(images, labels, count_per_class):
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]
    
        selected_0 = np.random.choice(idx_0, count_per_class, replace=False)
        selected_1 = np.random.choice(idx_1, count_per_class, replace=False)
        
        selected_idx = np.concatenate([selected_0, selected_1])
        np.random.shuffle(selected_idx)  
    
        return images[selected_idx], labels[selected_idx]
        
    # Balance each dataset
    poland_images_bal, poland_labels_bal = balance(poland_images, poland_labels, 100)
    china_images_bal, china_labels_bal = balance(china_images, china_labels, 200)
    egypt_images_bal, egypt_labels_bal = balance(egypt_images, egypt_labels, 100)

    # Print shape and label distributions
    print(f'Poland (balanced): images {poland_images_bal.shape}, labels {poland_labels_bal.shape}')
    print(f'China  (balanced): images {china_images_bal.shape}, labels {china_labels_bal.shape}')
    print(f'Egypt  (balanced): images {egypt_images_bal.shape}, labels {egypt_labels_bal.shape}')

    # Poland
    values, counts = np.unique(poland_labels_bal, return_counts=True)
    print("After balancing,")
    print("0 for malignant, 1 for benign,")
    print("Poland label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}\n")

    # China
    values, counts = np.unique(china_labels_bal, return_counts=True)
    print("China label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}\n")

    # Egypt
    values, counts = np.unique(egypt_labels_bal, return_counts=True)
    print("Egypt label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}")

    return (poland_images_bal, poland_labels_bal), (china_images_bal, china_labels_bal), (egypt_images_bal, egypt_labels_bal)

# full
def get_data_full_synth():
        # download and unzip the clean data 
    print('both synthesized and original data are here')
    file_id = '1nRx4cTtRnJMmRa-mCyouTAP44AVu0OZg'
    url = f'https://drive.google.com/uc?id={file_id}'   
    gdown.download(url, 'data_zipped_synth', quiet=False) 
    with zipfile.ZipFile('data_zipped_synth', 'r') as zipped_ref:
            zipped_ref.extractall('data_unzipped_synth')
    
    poland_images = np.load("/content/data_unzipped_synth/Datasets/Polish Hospital - Breast Ultrasound 128x128/Poland_Medical_Centers_US_Breast_Images.npy")
    poland_labels = np.load("/content/data_unzipped_synth/Datasets/Polish Hospital - Breast Ultrasound 128x128/Poland_Medical_Centers_US_Breast_Labels.npy")
    
    china_images = np.load("/content/data_unzipped_synth/Datasets/Chinese Hospitals - Kidney Ultrasound 128x128/Chinese Hospitals_Kidney_US_Images.npy")
    china_labels = np.load("/content/data_unzipped_synth/Datasets/Chinese Hospitals - Kidney Ultrasound 128x128/Chinese Hospitals_Kidney_US_Labels.npy")
    
    egypt_images = np.load("/content/data_unzipped_synth/Datasets/Egypt Hospital - Breast Ultrasound/Egypt Hospital 128x128/Egypt Hospital_US_Breast_Images.npy")
    egypt_labels = np.load("/content/data_unzipped_synth/Datasets/Egypt Hospital - Breast Ultrasound/Egypt Hospital 128x128/Egypt Hospital_US_Breast_Labels.npy")
    
    # 400, 100
    egypt_synth_CGAN_images = np.load('/content/data_unzipped_synth/Datasets/Nick_synth_egypt/images.npy')
    egypt_synth_CGAN_labels = np.load('/content/data_unzipped_synth/Datasets/Nick_synth_egypt/labels.npy')
    egypt_synth_WGAM_images = np.load('/content/data_unzipped_synth/Datasets/Saad_synth_egypt/images.npy')
    egypt_synth_WGA_labels = np.load('/content/data_unzipped_synth/Datasets/Saad_synth_egypt/labels.npy')
    
    
    def shuffle_data(images, labels):
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        return images[indices], labels[indices]
    
    poland_images, poland_labels = shuffle_data(poland_images, poland_labels)
    china_images, china_labels = shuffle_data(china_images, china_labels)
    egypt_images, egypt_labels = shuffle_data(egypt_images, egypt_labels)
    egypt_synth_CGAN_images, egypt_synth_CGAN_labels = shuffle_data(egypt_synth_CGAN_images, egypt_synth_CGAN_labels)
    egypt_synth_WGAM_images, egypt_synth_WGA_labels = shuffle_data(egypt_synth_WGAM_images, egypt_synth_WGA_labels)
    print('full dataset returned, with egypt synthesized of both CGAN, WGA, shuffled and unbalanced.')
    return (poland_images, poland_labels), (china_images, china_labels), (egypt_images, egypt_labels), (egypt_synth_CGAN_images,egypt_synth_CGAN_labels), (egypt_synth_WGAM_images,egypt_synth_WGA_labels)
    

def get_data_BIG_WGAN():
    print('downloading a file with egypt 1180 samples (Main + BIG-WGAN) and poland, china')
    file_id = '1PY29AhKzMo_msc0-8nc0oXnyWQHpbjWO'
    url = f'https://drive.google.com/uc?id={file_id}'   
    gdown.download(url, 'data_zipped', quiet=False) 
    with zipfile.ZipFile('data_zipped', 'r') as zipped_ref:
         zipped_ref.extractall('data_unzipped')
            
    poland_images = np.load("/content/data_unzipped/Datasets/Polish Hospital - Breast Ultrasound 128x128/Poland_Medical_Centers_US_Breast_Images.npy")
    poland_labels = np.load("/content/data_unzipped/Datasets/Polish Hospital - Breast Ultrasound 128x128/Poland_Medical_Centers_US_Breast_Labels.npy")
    
    china_images = np.load("/content/data_unzipped/Datasets/Chinese Hospitals - Kidney Ultrasound 128x128/Chinese Hospitals_Kidney_US_Images.npy")
    china_labels = np.load("/content/data_unzipped/Datasets/Chinese Hospitals - Kidney Ultrasound 128x128/Chinese Hospitals_Kidney_US_Labels.npy")
    
    egypt_images = np.load("/content/data_unzipped/Datasets/Egypt Hospital - Breast Ultrasound/Egypt Hospital 128x128/Egypt Hospital_US_Breast_Images.npy")
    egypt_labels = np.load("/content/data_unzipped/Datasets/Egypt Hospital - Breast Ultrasound/Egypt Hospital 128x128/Egypt Hospital_US_Breast_Labels.npy")
    
    egypt_BIG_WGAN_images = np.load("/content/data_unzipped/Datasets/GAN_images_rdy/gan_images_128x128_gray_shuffled.npy")
    egypt_BIG_WGAN_labels = np.load("/content/data_unzipped/Datasets/GAN_images_rdy/gan_labels_shuffled.npy")
    def shuffle_data(images, labels):
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        return images[indices], labels[indices]
        
    egypt_BIG_WGAN_images1 = egypt_BIG_WGAN_images.copy()
    egypt_BIG_WGAN_labels1 = egypt_BIG_WGAN_labels.copy()
    
    egypt_BIG_WGAN_images1 = egypt_BIG_WGAN_images1.squeeze()
    
    egypt_mixed_images = np.concatenate((egypt_images, egypt_BIG_WGAN_images1))
    egypt_mixed_labels = np.concatenate((egypt_labels, egypt_BIG_WGAN_labels1))
    
    egypt_mixed_images_1180, egypt_mixed_labels_1180 = shuffle_data(egypt_mixed_images, egypt_mixed_labels)
    
    flipped_images = np.flip(egypt_mixed_images_1180, axis=2)  # axis=2 is width
    # labels from images
    flipped_labels = egypt_mixed_labels_1180.copy()
    
    # flip and stick
    egypt_mixed_images_2360 = np.concatenate([egypt_mixed_images_1180, flipped_images], axis=0)
    egypt_mixed_labels_2360 = np.concatenate([egypt_mixed_labels_1180, flipped_labels], axis=0)
    
    
    poland_images, poland_labels = shuffle_data(poland_images, poland_labels)
    china_images, china_labels = shuffle_data(china_images, china_labels)
    egypt_images, egypt_labels = shuffle_data(egypt_images, egypt_labels)
        
        # check
    print(f'Poland: images {poland_images.shape}, labels {poland_labels.shape}')
    print(f'China: images {china_images.shape}, labels {china_labels.shape}')
    print(f'Egypt: images {egypt_images.shape}, labels {egypt_labels.shape}')
        
        # Poland
    values, counts = np.unique(poland_labels, return_counts=True)
    print("unbalanced, unshuffled,")
    print("0 for malignant, 1 for benign,")
    print("Poland label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}\n")
        
        # China
    values, counts = np.unique(china_labels, return_counts=True)
    print("China label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}\n")
        
        # Egypt
    values, counts = np.unique(egypt_labels, return_counts=True)
    print("Egypt label distribution:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}\n") 
    
    # egypt mixed with GAN 1180
    values, counts = np.unique(egypt_mixed_labels_1180, return_counts=True)
    print("Egypt mixed with GAN:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}\n") 
    
    values, counts = np.unique(egypt_mixed_labels_2360, return_counts=True)
    print("Egypt mixed with GAN and FLIPPED:")
    print(f"  Label {values[0]}: {counts[0]}")
    print(f"  Label {values[1]}: {counts[1]}") 
    
    print ('returns poland images(TEST THIS), china images, egypt images, egypt mixed images 1180(TRAIN THIS), egypt mixed images 2360(TRAIN THIS)')
    return (poland_images, poland_labels), (china_images, china_labels), (egypt_images, egypt_labels), (egypt_mixed_images_1180, egypt_mixed_labels_1180), (egypt_mixed_images_2360, egypt_mixed_labels_2360)
    
