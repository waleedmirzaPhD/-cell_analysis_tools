import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

class ImageBoundaryData:
    def __init__(self, base_path='./'):
        self.base_path = base_path
        self.boundary_pixels_data = {}
        self.junctions_data = {}
        self.tri_junctions_data = {}  # Stores tri-junctions for each image
        self.areas_by_gray_value = {}
    
    def compute_areas(self, img, pixel_resolution=1):
        """Compute areas for different grayscale values in the image."""
        gray_values = np.unique(img)[1:]  # Exclude the background
        for gray_value in gray_values:
            pixel_count = np.sum(img == gray_value)
            cell_area = pixel_count * pixel_resolution
            self.areas_by_gray_value[gray_value] = cell_area


    @staticmethod
    def is_boundary_pixel(y, x, img, neighbors, height, width):
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and img[y, x] != img[ny, nx]:
                return True
        return False

    def process_image(self, image_name, pixel_resolution=1):
        image_path = f"{self.base_path}{image_name}"
        img = io.imread(image_path)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        height, width = img.shape
        
        # Compute junctions and tri-junctions first
        junctions, tri_junctions = self.find_junctions(img, neighbors)
        self.junctions_data[image_name] = junctions
        self.tri_junctions_data[image_name] = tri_junctions

        # Then compute boundary pixels
        boundary_pixels_by_gray_value = {gray_value: [] for gray_value in np.unique(img)[1:]}  # Exclude background
        for gray_value in boundary_pixels_by_gray_value.keys():
            row_indices, col_indices = np.where(img == gray_value)
            boundary_pixels_by_gray_value[gray_value] = [(y, x) for y, x in zip(row_indices, col_indices) if self.is_boundary_pixel(y, x, img, neighbors, height, width)]
        
        self.boundary_pixels_data[image_name] = boundary_pixels_by_gray_value

    def process_images_in_range(self, start_index, end_index):
        for i in range(start_index, end_index + 1):
            image_name = f"tp{i:04}_00_cp_masks.tif"
            self.process_image(image_name)

    def get_boundary_pixels(self, image_name):
        return self.boundary_pixels_data.get(image_name, None)

    def visualize_image_with_boundaries(self, image_name):
        if image_name in self.boundary_pixels_data:
            image_path = f"{self.base_path}{image_name}"
            img = io.imread(image_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img, cmap='gray')
            
            boundary_pixels_by_gray_value = self.boundary_pixels_data[image_name]
            for gray_value, pixels in boundary_pixels_by_gray_value.items():
                if pixels:
                    y, x = zip(*pixels)
                    plt.scatter(x, y, s=1, label=f'Gray {gray_value}')
            
            plt.title(f'Image: {image_name} with Boundaries')
            plt.axis('off')
            plt.legend()
            plt.show()
        else:
            print(f"No boundary data found for image: {image_name}")
        
    def find_junctions(self, img, neighbors):
        junctions = []
        tri_junctions = []  # To store tri-junctions
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                neighbor_grays = set()
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                        neighbor_gray = img[ny, nx]
                        if neighbor_gray != img[y, x] and neighbor_gray != 0:
                            neighbor_grays.add(neighbor_gray)
                # First check for tri-junctions
                if len(neighbor_grays) >= 2:
                    tri_junctions.append((y, x))
                elif len(neighbor_grays) == 1:
                    junctions.append((y, x))
        return junctions, tri_junctions

    
    def visualize_image_with_boundaries_and_junctions(self, image_name):
        image_path = f"{self.base_path}{image_name}"
        img = io.imread(image_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img, cmap='gray')

        # Plot boundary pixels
        boundary_pixels_by_gray_value = self.boundary_pixels_data.get(image_name, {})
        for _, pixels in boundary_pixels_by_gray_value.items():
            y, x = zip(*pixels) if pixels else ([], [])
            plt.scatter(x, y, s=1, color='red', label='Boundary Pixels')

        # Plot junctions
        junctions = self.junctions_data.get(image_name, [])
        y_junc, x_junc = zip(*junctions) if junctions else ([], [])
        plt.scatter(x_junc, y_junc, s=10, color='yellow', label='Junctions')

        # Plot tri-junctions
        tri_junctions = self.tri_junctions_data.get(image_name, [])
        y_tri, x_tri = zip(*tri_junctions) if tri_junctions else ([], [])
        plt.scatter(x_tri, y_tri, s=30, color='blue', label='Tri-Junctions')

        plt.title(f'Image: {image_name}')
        plt.axis('off')
        plt.legend()
        plt.show()

def process_and_visualize_image(image_name, base_path):
    """
    Function to process and visualize a single image.
    This function is designed to be used with concurrent processing.
    """
    image_processor = ImageBoundaryData(base_path=base_path)
    image_processor.process_image(image_name)
    image_processor.visualize_image_with_boundaries_and_junctions(image_name)

def process_image_for_areas(image_name, base_path):
    image_processor = ImageBoundaryData(base_path=base_path)
    image_path = os.path.join(base_path, image_name)
    img = io.imread(image_path)
    image_processor.compute_areas(img)
    areas_by_gray_value = image_processor.areas_by_gray_value
    return areas_by_gray_value

def plot_areas_over_time(areas_over_time):
    plt.figure(figsize=(10, 6))
    for gray_value in sorted({key for areas in areas_over_time.values() for key in areas}):
        plt.plot([areas.get(gray_value, 0) for areas in areas_over_time.values()], label=f'Gray {gray_value}')
    
    plt.xlabel('Time Point')
    plt.ylabel('Area')
    plt.title('Area of Grayscale Values Over Time')
    plt.legend()
    plt.show()





if __name__ == "__main__":
    base_path = 'images/'  # Adjust to your images' directory
    start_index = 30
    end_index = 40
    
    # Determine the number of available processors
    num_processors = os.cpu_count()

    # Create a list of image names to process
    image_names = [f"tp{i:04}_00_cp_masks.tif" for i in range(start_index, end_index + 1)]
    
    # Use ProcessPoolExecutor to distribute the processing across available processors
    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        # Map the process_and_visualize_image function to each image name
        futures = [executor.submit(process_and_visualize_image, image_name, base_path) for image_name in image_names]
         

        # Optionally, wait for all futures to complete and handle results or exceptions
        for future in futures:
            try:
                future.result()  # This will re-raise any exception caught during processing
            except Exception as exc:
                print(f'Generated an exception: {exc}')


    areas_over_time = {}  # Dictionary to collect areas over time

    # Use ProcessPoolExecutor to distribute the processing across available processors
    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        # Create a future for each image processing task
        futures = {executor.submit(process_image_for_areas, name, base_path): name for name in image_names}
        
        # Wait for all futures to complete and process results
        for future in as_completed(futures):
            image_name = futures[future]
            try:
                areas_by_gray_value = future.result()
                areas_over_time[image_name] = areas_by_gray_value
            except Exception as exc:
                print(f'{image_name} generated an exception: {exc}')

    plot_areas_over_time(areas_over_time)
