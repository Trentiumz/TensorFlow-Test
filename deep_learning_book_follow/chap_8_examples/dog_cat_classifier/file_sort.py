import shutil, pathlib, os

orig_dir = pathlib.Path("dogs-vs-cats/train/train")
new_dir = pathlib.Path("dogs-vs-cats/small")

def make_subset(name, start_ind, end_ind):
    for category in ("cat", "dog"):
        in_dir = new_dir / name / category
        os.makedirs(in_dir)
        o_files = [f"{category}.{i}.jpg" for i in range(start_ind, end_ind)]
        n_files = [f"{i}.jpg" for i in range(start_ind, end_ind)]

        for o_name, n_name in zip(o_files, n_files):
            shutil.copyfile(src=orig_dir / o_name, dst = in_dir / n_name)

make_subset("train", 0, 1000)
make_subset("validation", 1000, 1500)
make_subset("test", 1500, 2500)