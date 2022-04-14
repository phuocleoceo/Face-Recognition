from os import listdir, rename

if __name__ == "__main__":
    # listPeople = listdir(f"./People/")

    listPeople = ["Huan", "Quan", "DucBear", "Hung", "DucTruong"]
    for p in listPeople:
        listImg = listdir(f"./People/{p}")
        for i, img in enumerate(listImg):
            rename(f"./People/{p}/{img}", f"./People/{p}/{i+1}.jpg")
