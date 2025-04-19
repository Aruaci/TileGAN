import os
import glob

def get_img_paths(
        paths,
        subfolder="train", 
        filending="png"
        ): 
     
    categories = ["crack", "glue_strip", "gray_stroke", "oil", "rough", "good"]
    pattern = "*" + filending
    
    for c in categories:
        path = "./tile/" + subfolder + "/" + c
        search_path = os.path.join(path, pattern)
        found_files = glob.glob(search_path)
        paths[c].extend(found_files)

    return paths