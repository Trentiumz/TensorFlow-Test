import os, shutil

# define constants
orig_dir = "./PetImages"

base_dir = "./Data"
os.mkdir(base_dir)

train_dir = base_dir + "/Train"
validation_dir = base_dir + "/Val"
test_dir = base_dir + "/Test"

orig_cat = orig_dir + "/Cat"
orig_dog = orig_dir + "/Dog"
num_cats = 12500
num_dogs = 12500

train_to = 0.8
val_to = 0.9
test_to = 1

# start processing
cat_start = 0
dog_start = 0
for to_dir, to in ((train_dir, train_to), (validation_dir, val_to), (test_dir, test_to)):
    os.mkdir(to_dir)
    os.mkdir(to_dir + "/Cat")
    os.mkdir(to_dir + "/Dog")

    for ind in range(cat_start, int(num_cats * to)):
        shutil.copy(orig_cat + f"/{ind}.jpg", to_dir + f"/Cat/{ind}.jpg")
        cat_start = ind + 1
        if ind % 1000 == 0:
            print(f"cat {ind}")
    for ind in range(dog_start, int(num_dogs * to)):
        shutil.copy(orig_dog + f"/{ind}.jpg", to_dir + f"/Dog/{ind}.jpg")
        dog_start = ind + 1
        if ind % 1000 == 0:

            print(f"dog {ind}")
