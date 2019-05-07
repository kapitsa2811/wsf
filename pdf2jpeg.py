import pdf2image
from pdf2image import convert_from_path
import os


dirPath='C:\\Users\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval\query\pdf\\'

f=["Civic Education and Knowledge of Government and Politics.pdf","Economics for Everyone.pdf","History of Economic Thought - Modern Economics.pdf",
   "PID56821031.pdf","Principles_of_Economics.pdf","Textbook of Engineering Drawing.pdf","sp46.pdf"]
fileName=f[6]

if not os.path.isdir(dirPath+fileName[:-4]):
    os.mkdir(dirPath+fileName[:-4])

dumpPath=dirPath+fileName[:-4]

'''
pages = convert_from_path(dirPath+fileName,500)

for page in pages:
    page.save('out.jpg', 'JPEG')
'''

from wand.image import Image

with(Image(filename=dirPath+fileName, resolution=120)) as source:
    images = source.sequence
    pages = len(images)
    for i in range(pages):
        n = i + 1
        newfilename = fileName[:-4] + str(n) + '.jpeg'
        Image(images[i]).save(filename=dumpPath+newfilename)
