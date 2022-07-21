LR_path = 'LR image'
root_path = 'root path for offset-sr and offset (obtain from SRPO)'
save_root = "Set yours"
flat_choice = 'Lanczos' # Lanczos or Bilinear

# Load and process all images
for idx in range(9):
    # Load Offset and corresponding Offset SR
    offset_path = root_path + "/offset/{}.npy".format(idx)
    offset_sr_path = root_path + "/offset_H/{}.png".format(idx)
    lr_path = "{}/{}.png".format(save_root, idx)

    # lr | offset | offset_sr | 
    lr_img = Image.open(lr_path)
    offset_sr = Image.open(offset_sr_path)
    offsets = np.load(offset_path)

    # lr -> flat_sr
    new_width, new_height = offset_sr.size
    if flat_choice == 'Lanczos':
        flat_sr = lr_img.resize((new_width, new_height), Image.LANCZOS)
    elif flat_choice == 'Bilinear':
        flat_sr = lr_img.resize((new_width, new_height), Image.BILINEAR)
    else:
        raise NotImplementedError('Wrong Falt Choice')

    # to tensor
    offset_sr_pth = to_tensor(offset_sr).unsqueeze(0)
    flat_sr_pth = to_tensor(flat_sr).unsqueeze(0)

    # offset -> mask -> mask_blur
    offsets = np.load(offset_path)
    offsets_over_1 = np.abs(np.round(offsets)).max(0).clip(0,1)
    offsets_blurred = to_tensor(gaussian_filter(offsets_over_1, sigma=0.5))

    blended = offsets_blurred * offset_sr_pth + (1-offsets_blurred) * flat_sr_pth
    result = to_pil_image(blended.squeeze(0))
    result.save(save_root + '/blendsr_{}_{}.png'.format(flat_choice, idx))