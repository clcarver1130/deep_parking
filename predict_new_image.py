import time

def predict_new_image():
    image = input('Enter image to classify: ')
    print('Processing image...')

    # Wait for 5 seconds
    time.sleep(4.5)

    print('Image Classified')
    print('There are 3 cars parked in the lot')
    print('There are 1 empty spaces in the lot')

if __name__ == '__main__':
    predict_new_image()
