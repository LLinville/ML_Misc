from PIL import Image
import numpy as np
from vgg_face.vgg_model import vgg_vd_face_fer_dag
import torch

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

if __name__ == "__main__":
    model = vgg_vd_face_fer_dag("/Users/levi.linville/Downloads/vgg_vd_face_fer_dag.pth")
    image1_data = load_image("/Users/levi.linville/Desktop/screenshots/Screen Shot 2019-05-23 at 3.22.02 PM.png")
    image2_data = load_image("/Users/levi.linville/Desktop/screenshots/Screen Shot 2019-04-24 at 4.28.39 PM.png")
    image2_padded = np.zeros_like(image1_data)
    image2_padded[:image2_data.shape[0], :image2_data.shape[1]] = image2_data
    data = np.stack((image1_data, image2_padded))
    data = np.transpose(data, (0, 3, 1, 2))
    data = data[:,:3,...]
    data = torch.Tensor(data)
    model.forward(data)
    print("loaded")