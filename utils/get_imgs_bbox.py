import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt



def load_data(path):
    df = pd.read_csv(path)
    return df


def get_data(data, to_idx):
    
    data_new = data[:to_idx]   
    file_names = data_new['File_name'].tolist()  
    bboxes = data_new['Bounding_boxes'].tolist()
    
    bboxes = [f.replace(' ','').split(',') for f in bboxes]
    bboxes_v=[]
    for box in bboxes:
        b = []
        for v in box:
            v = float(v)
            v = int(v)
            b.append(v)
        bboxes_v.append(b)
    # bboxes_v = [float(v) for v in f for f in bboxes]
    
    root_dir = '/Users/mustafa/Documents/GitHub/DATA/DeepLesion/data/Images_png'
    files_path = [root_dir+'/'+'_'.join(f.split('_')[0:3])+'/'+f.split('_')[-1] for f in file_names]
    
    return files_path , bboxes_v

# def get_paths(files):
#     root_dir = '/Users/mustafa/Documents/GitHub/DATA/DeepLesion/data/Images_png'
#     paths = [root_dir+'/'+'_'.join(f.split('_')[0:3])+'/'+f.split('_')[-1] for f in files]
    
#     return paths
    
    



def plot_images_with_boxes(images, bounding_boxes, ncols=None):
    """
    Plot a grid of images with bounding boxes.

    Args:
    images (list of numpy arrays): List of images to be plotted.
    bounding_boxes (list of lists of tuples): List of bounding boxes for each image.
    Each bounding box is represented as a list of tuples (x1, y1, x2, y2).
    ncols (int, optional): Number of columns in the grid. If None, the number of columns is automatically determined.

    Returns:
    None
    """
    num_images = len(images)
    if num_images == 0:
        return

    if ncols is None:
        ncols = int(np.ceil(np.sqrt(num_images)))

    nrows = int(np.ceil(num_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            ax.axis('off')
            ax.set_aspect('equal')

            index = i * ncols + j

            if index < num_images:
                image = images[index]
                boxes = bounding_boxes[index]
                ax.imshow(image)
                x1, y1, x2, y2 = boxes
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', lw=2))
                # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2) 

    for i in range(num_images, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()
    
    
if __name__=="__main__":
    
    df = load_data('../csv/DL_info.csv')
    
    file_paths ,bboxes = get_data(df,20)

    
    

    images = [cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in file_paths]
    # print(images)
    
    plot_images_with_boxes(images, bboxes, ncols=None)