from sort_file_and_clean_filename import sort_and_clean_filenames
from crop_images import crop_images_with_trained_YOLO
from normalize_images import normalize_images
from predict_inner_outer_masks import predict_inner_outer_masks
from apply_the_inner_outer_masks import get_masked_images
from get_mean_intensity import get_mean_intensity


if __name__ == "__main__":
    # Set the input and output folders
    input_folder = r'C:\Users\yifei\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\test\Plate 1'
    
    # Call the function to sort and clean the filenames
    output_folder = sort_and_clean_filenames(input_folder)

    # crop the center of the images out
    crop_images_with_trained_YOLO(output_folder)

    # Normalize the images
    normalize_images(output_folder)

    # Predict inner and outer masks
    predict_inner_outer_masks(output_folder)

    # Process images with corresponding outer and inner masks
    get_masked_images(output_folder)

    # Get mean intensity
    get_mean_intensity(output_folder)










