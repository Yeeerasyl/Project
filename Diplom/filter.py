import os
import shutil
from random import shuffle

def split_dataset(dataset_path, output_path, train_ratio=0.8):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Получаем список папок с животными внутри папки dataset
    animal_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    for animal_folder in animal_folders:
        animal_path = os.path.join(dataset_path, animal_folder)

        # Получаем список фотографий внутри папки с животным
        photos = [f for f in os.listdir(animal_path) if f.endswith(('.jpg', '.jpeg', '.png'))]


        shuffle(photos)

        # Вычисляем количество фотографий для train и test
        num_train = int(len(photos) * train_ratio)
        train_photos = photos[:num_train]
        test_photos = photos[num_train:]

        # Создаем папки train и test внутри новой папки для текущего животного
        train_path = os.path.join(output_path, 'train', animal_folder)
        test_path = os.path.join(output_path, 'test', animal_folder)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for photo in train_photos:
            shutil.copy(os.path.join(animal_path, photo), os.path.join(train_path, photo))

        for photo in test_photos:
            shutil.copy(os.path.join(animal_path, photo), os.path.join(test_path, photo))

if __name__ == "__main__":
    dataset_path = "C:/Users/yeras/OneDrive/Рабочий стол/Diplom/dataset"
    output_path = "C:/Users/yeras/OneDrive/Рабочий стол/Diplom/output"

    split_dataset(dataset_path, output_path)

