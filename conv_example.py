img = np.asarray(Image.open('./results/result.png'))
img2 = rgb2ycbcr(img)
img3 = Image.fromarray(ycbcr2rgb(img2))

img_f = Image.fromarray(img)
img3.save('my_conversion.png')
'''
'''
ycbcr_img_origin = Image.fromarray(img).convert('YCbCr')
ycbcr_img_origin_arr = np.asarray(ycbcr_img_origin)
ycbcr_img = Image.fromarray(ycbcr_img_origin_arr).convert('RGB')
ycbcr_img.save('PIL_conversion.png')

