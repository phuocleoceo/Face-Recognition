from inception_resnet_v1 import InceptionResNetV1
from os.path import join
from os import listdir
import numpy as np

if __name__ == "__main__":
    WEIGHT_PATH = "./model_weights"
    # Load kiến trúc Model
    model = InceptionResNetV1()
    # Load weights layer trong thư mục weights
    layer_files = listdir(WEIGHT_PATH)
    # Duyệt các layer của model Inception resnet
    for layer in model.layers:
        # Lấy những file weight thuộc layer đang xét
        weight_files = [l for l in layer_files if l.split(".")[0] == layer.name]
        for wf in weight_files:
            # Load file weight
            weight_loaded = np.load(join(WEIGHT_PATH, wf))
            weights_for_layer = []
            for wl in weight_loaded:
                weights_for_layer.append(weight_loaded[wl])
            try:
                layer.set_weights(weights_for_layer)
            except:
                pass

    print("-----------------------------Input-----------------------------")
    print(model.input)
    print("-----------------------------Output----------------------------")
    print(model.output)
    print("---------------------------Save model--------------------------")
    model.save("./FaceNet_Keras.h5")
    print(">> Model saved in FaceNet_Keras.h5")
