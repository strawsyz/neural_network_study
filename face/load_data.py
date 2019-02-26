from PIL import Image
import numpy


def load_data(dataset_path):  
    img = Image.open(dataset_path)  
    # 数据归一化
    img_ndarray = numpy.asarray(img, dtype='float64')/256  
    faces=numpy.empty((400,2679))  
    # 把20X20张的57X47大小的图片
    # 放到faces中57x47个像素的数据放在一行
    # 共有400行，即400条数据
    for row in range(20):  
       for column in range(20):  
        faces[row*20+column] = numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])  
    
    label=numpy.empty(400)  
    # 40个人的头像，每人10张
    # 共40个标签，每类标签10张图片
    for i in range(40): 
        label[i*10:i*10+10] = i
        label = label.astype(numpy.int)
  
    # 8：1:1分成训练、验证、测试集
    train_data = numpy.empty((320,2679))  
    train_label = numpy.empty(320)  
    valid_data = numpy.empty((40,2679))  
    valid_label = numpy.empty(40)  
    test_data = numpy.empty((40,2679))  
    test_label = numpy.empty(40)  
  
    for i in range(40):  
        train_data[i*8:i*8+8] = faces[i*10:i*10+8]  
        train_label[i*8:i*8+8] = label[i*10:i*10+8]  
        valid_data[i] = faces[i*10+8]  
        valid_label[i] = label[i*10+8]  
        test_data[i] = faces[i*10+9]  
        test_label[i] = label[i*10+9]  

    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')
  
    result = [(train_data, train_label), (valid_data, valid_label), (test_data, test_label)]
    return result

# temp = load_data('olivettifaces.gif')
# print(temp)