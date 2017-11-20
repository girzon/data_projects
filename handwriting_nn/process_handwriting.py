from PIL import Image
import numpy as np
import sys

offset = int((32/8) * 4)
label_offset = int((32/8) * 2)
image_length = int(28)
image_size = (image_length*image_length) 
label_size = int(1) 


def write_image(img_no,data,sub_folder):

    im = Image.new('1', (image_length, image_length))
    pixels = im.load()
    #replace with write array
    for i in range(0,image_length-1):
        for j in range(0,image_length -1):
            pixels[i,j] = data[(j*image_length)+i]

    #img.show()
    im.save(sub_folder+"img/char_"+str(img_no)+".png")

def value_encode(img_num):
    img_array=[0]*10
    img_array[img_num] = 1  
    return img_array   

def process_file(image_file,label_file,sub_folder):
    #open files
    image_bytes_read = open(image_file, "rb").read()
    

    #read Images data
    #split metadata and data
    data_image = image_bytes_read[offset:len(image_bytes_read)]
    meta_data = image_bytes_read[0:offset-1]
    #data2 = int.from_bytes(image_bytes_read[0:4], byteorder='big') # metadata
    #meta_data = data2

    #load images into array
    data_image_int = [int(data_image[s]) for s in range(0, len(data_image))]
    img_list = [data_image_int[s:s + image_size] for s in range(0, len(data_image_int), image_size)]

    #read label data
    #split metadata and data
    label_bytes_read = open(label_file, "rb").read()
    data_label = label_bytes_read[label_offset:len(label_bytes_read)]

    #load images into array
    data_label_int = [int(data_label[s]) for s in range(0, len(data_label))]
    label_list = [data_label_int[s] for s in range(0, len(data_label_int),label_size)]
    label_list_encode = [value_encode(data_label_int[s]) for s in range(0, len(data_label_int),label_size)]

    #Write Data out
    #Create images
    for index,img_data in enumerate(img_list):
        alabel = 'result_'+str(label_list[index])+'_no_'+str(index)
        write_image(alabel,img_data,sub_folder)

    #Create csv for images (input)
    heading_names = ",".join(['I'+str(s+1) for s in range(0, image_size)])
    np_export = np.asarray(img_list)
    np.savetxt(sub_folder+"imagedata.csv",np_export, fmt=['%i']*image_size, delimiter=",",header = heading_names,comments='')

    #Create csv for labels (output)
    heading_names = (",".join(['O'+str(s) for s in range(0, 10)]))
    np_export_label = np.asarray(label_list_encode)
    np.savetxt(sub_folder+"labeldata.csv",np_export_label, fmt=['%i']*10, delimiter=",",header = heading_names,comments='')
  

def main():
    process_file("./bin_data/train-images.idx3-ubyte","./bin_data/train-labels.idx1-ubyte",'./train_data/')
    process_file("./bin_data/t10k-images.idx3-ubyte","./bin_data/t10k-labels.idx1-ubyte",'./test_data/')

if __name__ == "__main__":
    main()