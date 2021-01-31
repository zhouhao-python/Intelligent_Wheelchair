from PIL import Image

txt_path = './labels.txt'

with open(txt_path, 'r') as f:
    img_nums = []
    for line in f:
        image_list = [img_path for img_path in line.split(',')]
        img_nums.append((image_list[:-1],image_list[-1]))
print(img_nums[0][0][0])
image = Image.open(img_nums[0][0][0]).convert('RGB')
print(image)



# print(image_list)
# print(img_nums)
# print(len(img_nums))