from DeeplabV2_resnet101 import ResNet101
from myMetrics import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import argparse

n_class = 108

print("load model weights")
deeplab_model = ResNet101(input_shape=(None, None, 3), classes=108)

parser = argparse.ArgumentParser()
parser.add_argument("--check_point", type=str, default='not_defined')
parser.add_argument("--path", type=str, default='not_defined')
args = parser.parse_args()

pathCP = args.check_point
pathSave = args.path

assert not (pathCP == 'not_defined'), "Checkpoint path not defined"
if pathSave == 'not_defined':
    pathSave = "./"
else:
    pathSave = pathSave + "/"

pT = pathSave + datetime.now().strftime("%Y%m%d-%H%M%S")
print(pathCP)
if not os.path.isdir(pT):
    os.mkdir(pT)
inf_path = pT + "/inf"
os.mkdir(inf_path)


deeplab_model.load_weights(pathCP, by_name=True)

dir_img = "D:/Rossi/data/val/"
dir_seg = "D:/Rossi/data_part_107part_val/"
####

images = glob.glob(dir_img + "*.png")
images.sort()
segs = glob.glob(dir_seg + "*.png")
segs.sort()

pathCMap = "Y:/tesisti/rossi/cmap255.mat"
fileMat = loadmat(pathCMap)
cmap = fileMat['cmap']

mat = np.zeros(shape=(n_class, n_class), dtype=np.int32)

for k in tqdm(range(len(segs))):
    img = cv2.imread(images[k])
    seg = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)

    w, h, _ = img.shape
    mask = seg != 255
    seg = seg[mask]
    scale = img / 127.5 - 1

    res = deeplab_model.predict(np.expand_dims(scale, 0))
    labels = np.argmax(res.squeeze(), -1)
    labels = labels.astype(np.uint8)
    labels_mask = labels[mask]

    tmp = confusion_matrix(seg, labels_mask, range(n_class))
    mat = mat + tmp

    # TO color labels and save image

    h_z, w_z = labels.shape
    imgNew = np.zeros((h_z, w_z, 3), np.uint8)
    for i in range(0, n_class):
        mask = cv2.inRange(labels, i, i)
        v = cmap[i]
        imgNew[mask > 0] = v

    name = images[k]
    name = name[-15:]

    pathTmp = inf_path + "/" + name
    cv2.imwrite(pathTmp, imgNew)

listParts = listPartsNames()
mIoU, IoU_per_class = compute_and_print_IoU_per_class(confusion_matrix=mat, num_classes=n_class, namePart=listParts)
create_excel_file(results108=IoU_per_class, path=pT + "/")
print(mIoU)
