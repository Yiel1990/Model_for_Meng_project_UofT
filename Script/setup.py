import os

os.system("cat ./VGG_model/Model.* > ./VGG_model/Model.356-0.6796.hdf5.tar.gz")
os.system("tar xvzf ./VGG_model/Model.356-0.6796.hdf5.tar.gz -C ./VGG_model")
os.system("rm ./VGG_model/Model.356-0.6796.hdf5.tar.gz")

os.system("cat ./VGG_19_model/VGG_19_model.* > ./VGG_19_model/VGG_19_Model.310-0.6668.hdf5.tar.gz")
os.system("tar xvzf ./VGG_19_model/VGG_19_Model.310-0.6668.hdf5.tar.gz -C ./VGG_19_model")
os.system("rm ./VGG_19_model/VGG_19_Model.310-0.6668.hdf5.tar.gz")
