import cv2
import matplotlib.pyplot as plt
from utils import measure_diameter

def measure_silicon_diameter(path, lumerical_nm= None, reference_px= None):

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    diameter_px = measure_diameter(binary_matrix= binary / 255)
    print(f"Diameter (pixels): {diameter_px}")

    if lumerical_nm != None and reference_px != None:
        print(f"Diameter (nm): {diameter_px * (lumerical_nm / reference_px):.4f}")

    _, ax = plt.subplots(1,2, figsize= (12,6))
    ax[0].imshow(gray, cmap= 'gray')
    ax[0].axis('off')
    ax[0].set_title('Grayscale')
    ax[1].imshow(binary, cmap= 'gray')
    ax[1].axis('off')
    ax[1].set_title('Binary (Otsu)')
    plt.show()

    return diameter_px

if __name__ == '__main__':
    # first function call to find the diameter of the original silicon
    reference_px= measure_silicon_diameter(path=r'C:\Vscode_Python\Pixel_Mapping\silicons\cropped_fig28_.740113.jpg') # use the original silicon img path

    # second function call to apply the found diameter value and lumerical_nm value in order to get diameter in nm
    measure_silicon_diameter(path=r'C:\Vscode_Python\Pixel_Mapping\silicons\cropped_generated_0.png', # use the generated silicon img path
                             reference_px=reference_px,
                             lumerical_nm=5) # physical diameter of the real silicon hole as designed in Lumerical (in nm)
