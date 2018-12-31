# kích thước grid system
cell_size = 7
# số boundary box cần dự đoán mỗi ô vuông
box_per_cell = 2
# kích thước ảnh đầu vào
img_size = 224
# số loại nhãn
classes = {'circle': 0, 'triangle': 1, 'rectangle': 2}
nclass = len(classes)

box_scale = 5.0
noobject_scale = 0.5
batch_size = 128
# số lần huấn luyện
# thử thay đổi số lần huấn luyện để cho kết quả tốt hơn
epochs = 100
# learning của chúng ta
lr = 1e-3